# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``agilerl.utils.segment_balance``.

The cross-rank tests spawn two gloo ranks, which exchange rows on CPU.
"""

from __future__ import annotations

import os
import socket
import sys
import traceback
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from agilerl.utils.segment_balance import balance_rows_across_ranks, balanced_row_plan
from agilerl.utils.segment_rows import SegmentRows, split_episode_segments
from tests.test_utils.test_segment_rows import PAD, split_small_batch

requires_gloo = pytest.mark.skipif(
    sys.platform == "win32" or not dist.is_available(), reason="gloo unavailable"
)


def rank_one_rows() -> SegmentRows:
    """One unsegmented ``T = 6`` episode with 5 real tokens and one vision row."""
    return split_episode_segments(
        torch.tensor([[11, 12, 13, 14, 15, PAD]]),
        torch.tensor([[False, True, True, False, False]]),
        [None],
        PAD,
        turn_ids=torch.tensor([[-1, 0, 0, -1, -1]]),
        sampling_logps=[torch.tensor([-0.7, -0.8])],
        pixel_values=torch.tensor([[20.0, 21.0]]),
        pixel_image_counts=[1],
    )


def four_long_rows() -> SegmentRows:
    """Four unsegmented ``T = 11`` episodes with 10, 9, 8 and 7 real tokens.

    Each has two action tokens and one vision row ``[2e, 2e + 1]``; episode 3
    has no sampling log-probs.
    """
    lengths = [10, 9, 8, 7]
    token_ids = torch.full((4, 11), PAD)
    for episode, length in enumerate(lengths):
        token_ids[episode, :length] = torch.arange(length) + 10 * (episode + 1)
    action_masks = torch.zeros(4, 10, dtype=torch.bool)
    action_masks[:, 1:3] = True
    return split_episode_segments(
        token_ids,
        action_masks,
        [None] * 4,
        PAD,
        turn_ids=torch.where(action_masks, 0, -1),
        sampling_logps=[
            torch.tensor([-0.1, -0.2]),
            torch.tensor([-0.3, -0.4]),
            torch.tensor([-0.5, -0.6]),
            None,
        ],
        pixel_values=torch.arange(8.0).reshape(4, 2),
        pixel_image_counts=[1, 1, 1, 1],
    )


def row_values(rows: SegmentRows, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Distinct per-token advantages and episode token counts for each row."""
    num_rows, width = rows.action_masks.shape
    advantages = torch.arange(num_rows * width, dtype=torch.float32).reshape(
        num_rows, width
    ) + 100.0 * (rank + 1)
    episode_tokens = torch.arange(num_rows, dtype=torch.float32) + 10.0 * (rank + 1)
    return advantages, episode_tokens


def group_times(
    costs: list[list[int]], plan: list[list[tuple[int, int]]], ranks_per_group: int
) -> list[int]:
    """Each lockstep group's time: the sum over rounds of its costliest row."""
    times = []
    for first in range(0, len(plan), ranks_per_group):
        members = plan[first : first + ranks_per_group]
        rounds = max(len(rows) for rows in members)
        times.append(
            sum(
                max(
                    costs[source][row]
                    for rows in members
                    if position < len(rows)
                    for source, row in [rows[position]]
                )
                for position in range(rounds)
            )
        )
    return times


def balance_worker(rank: int, world_size: int, port: int, queue: mp.Queue) -> None:
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        rows = split_small_batch() if rank == 0 else rank_one_rows()
        advantages, episode_tokens = row_values(rows, rank)

        balanced = balance_rows_across_ranks(
            rows,
            np.arange(len(rows.row_episodes)),
            [(advantages, 0.0), (episode_tokens, 0.0)],
            PAD,
            None,
        )

        assert balanced is not None
        balanced, (dealt_advantages, dealt_tokens) = balanced
        if rank == 0:
            # Rank 0 keeps its 4-token row 2, then its 3-token row 0.
            assert balanced.token_ids.tolist() == [
                [7, 8, 1, 2, PAD, PAD],
                [1, 2, 3, PAD, PAD, PAD],
            ]
            assert balanced.row_episodes.tolist() == [1, 0]
            assert balanced.row_starts.tolist() == [0, 0]
            assert balanced.row_ends.tolist() == [6, 3]
            assert balanced.pixel_image_counts == [2, 1]
            assert balanced.pixel_values.tolist() == [[6, 7], [8, 9], [0, 1]]
            assert torch.equal(dealt_advantages[0], advantages[2])
            assert dealt_advantages[1].tolist() == [100.0, 101.0, 0, 0, 0]
            assert dealt_tokens.tolist() == [12.0, 10.0]
        else:
            # Rank 1 keeps its 5-token row, then takes rank 0's row 1.
            assert balanced.token_ids.tolist() == [
                [11, 12, 13, 14, 15, PAD],
                [4, 5, 6, PAD, PAD, PAD],
            ]
            assert balanced.row_episodes.tolist() == [2, 0]
            assert balanced.row_starts.tolist() == [0, 3]
            assert balanced.row_ends.tolist() == [6, 6]
            assert balanced.action_masks.tolist() == [
                [False, True, True, False, False],
                [False, True, False, False, False],
            ]
            assert balanced.action_masks.dtype == torch.bool
            assert balanced.turn_ids.tolist() == [
                [-1, 0, 0, -1, -1],
                [-1, 1, -1, -1, -1],
            ]
            assert [logps.tolist() for logps in balanced.sampling_logps] == [
                pytest.approx([-0.7, -0.8]),
                pytest.approx([-0.2]),
            ]
            assert balanced.pixel_image_counts == [1, 2]
            assert balanced.pixel_values.tolist() == [[20, 21], [2, 3], [4, 5]]
            assert dealt_advantages[1].tolist() == [105.0, 106.0, 0, 0, 0]
            assert dealt_tokens.tolist() == [20.0, 11.0]
        queue.put((rank, "ok", ""))
    except BaseException:
        queue.put((rank, "fail", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def values_worker(rank: int, world_size: int, port: int, queue: mp.Queue) -> None:
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        rows = split_small_batch() if rank == 0 else rank_one_rows()
        num_rows, width = rows.action_masks.shape
        codes = torch.arange(num_rows * width).reshape(num_rows, width) + 1000 * (
            rank + 1
        )
        pairs = torch.arange(num_rows * 2, dtype=torch.float64).reshape(
            num_rows, 2
        ) + 10 * (rank + 1)

        balanced = balance_rows_across_ranks(
            rows,
            np.arange(len(rows.row_episodes)),
            [(codes, -1), (pairs, 0.0)],
            PAD,
            None,
        )

        assert balanced is not None
        _rows, (dealt_codes, dealt_pairs) = balanced
        assert dealt_codes.dtype == torch.long
        assert dealt_pairs.dtype == torch.float64
        if rank == 0:
            assert dealt_codes.tolist() == [
                [1010, 1011, 1012, 1013, 1014],
                [1000, 1001, -1, -1, -1],
            ]
            assert dealt_pairs.tolist() == [[14.0, 15.0], [10.0, 11.0]]
        else:
            assert dealt_codes.tolist() == [
                [2000, 2001, 2002, 2003, 2004],
                [1005, 1006, -1, -1, -1],
            ]
            assert dealt_pairs.tolist() == [[20.0, 21.0], [12.0, 13.0]]
        queue.put((rank, "ok", ""))
    except BaseException:
        queue.put((rank, "fail", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def devices_worker(
    rank: int, world_size: int, port: int, queue: mp.Queue, device: str
) -> None:
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        rows = four_long_rows() if rank == 0 else rank_one_rows()
        # Sampling log-probs stay on CPU, as rollouts hand them to learn.
        rows = replace(
            rows,
            token_ids=rows.token_ids.to(device),
            action_masks=rows.action_masks.to(device),
            turn_ids=rows.turn_ids.to(device),
            pixel_values=rows.pixel_values.to(device),
        )
        advantages, episode_tokens = row_values(rows, rank)

        balanced = balance_rows_across_ranks(
            rows,
            np.arange(len(rows.row_episodes)),
            [(advantages, 0.0), (episode_tokens, 0.0)],
            PAD,
            None,
        )

        assert balanced is not None
        balanced, _values = balanced
        assert balanced.pixel_values.device.type == torch.device(device).type
        if rank == 0:
            assert balanced.row_episodes.tolist() == [0, 2]
            assert balanced.pixel_values.tolist() == [[0, 1], [4, 5]]
        else:
            # Rank 1 takes rank 0's rows 1 and 3, then keeps its own row.
            assert balanced.row_episodes.tolist() == [1, 3, 4]
            assert balanced.pixel_values.tolist() == [[2, 3], [6, 7], [20, 21]]
            logps = balanced.sampling_logps
            assert logps[0].tolist() == pytest.approx([-0.3, -0.4])
            assert logps[1] is None
            assert logps[2].tolist() == pytest.approx([-0.7, -0.8])
        queue.put((rank, "ok", ""))
    except BaseException:
        queue.put((rank, "fail", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def too_few_rows_worker(rank: int, world_size: int, port: int, queue: mp.Queue) -> None:
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        rows = rank_one_rows()
        advantages, episode_tokens = row_values(rows, rank)
        train_rows = np.arange(1) if rank == 0 else np.arange(0)

        balanced = balance_rows_across_ranks(
            rows, train_rows, [(advantages, 0.0), (episode_tokens, 0.0)], PAD, None
        )

        assert balanced is None
        queue.put((rank, "ok", ""))
    except BaseException:
        queue.put((rank, "fail", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def spawn_ranks(
    worker, *args: object, world_size: int = 2, timeout: float = 120.0
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=worker, args=(rank, world_size, port, queue, *args))
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    results = [queue.get(timeout=timeout) for _ in range(world_size)]
    for proc in procs:
        proc.join(timeout=timeout)
    for rank, status, error in sorted(results):
        assert status == "ok", f"rank {rank}: {error}"


class TestBalancedRowPlan:
    def test_every_rank_runs_the_same_number_of_rows(self) -> None:
        # Arrange: rank 0 has five rows, rank 1 one.
        costs = [[30, 30, 30, 30, 10], [5]]

        # Act
        plan = balanced_row_plan(costs, len(costs))

        # Assert: the i-th rows of both ranks are as alike as the rows allow.
        assert plan == [
            [(0, 0), (0, 2), (0, 4)],
            [(0, 1), (0, 3), (1, 0)],
        ]

    def test_rows_stay_on_their_rank_when_ranks_are_already_even(self) -> None:
        # Arrange
        costs = [[9, 1], [2, 8]]

        # Act
        plan = balanced_row_plan(costs, len(costs))

        # Assert
        assert plan == [[(0, 0), (0, 1)], [(1, 1), (1, 0)]]

    def test_a_partial_last_round_leaves_some_ranks_a_row_short(self) -> None:
        # Arrange
        costs = [[4, 3, 2], [], [1]]

        # Act
        plan = balanced_row_plan(costs, len(costs))

        # Assert
        assert plan == [[(0, 0)], [(0, 1)], [(0, 2), (2, 0)]]

    def test_lockstep_groups_finish_together(self) -> None:
        # Arrange: ranks 0 and 1 hold every costly row; groups are {0, 1}, {2, 3}.
        costs = [[10, 8, 6], [9, 7, 5], [1, 1, 1], [1, 1, 1]]

        # Act
        grouped = balanced_row_plan(costs, 2)
        world = balanced_row_plan(costs, 4)

        # Assert
        assert [len(rows) for rows in grouped] == [3, 3, 3, 3]
        assert group_times(costs, grouped, 2) == [16, 16]
        assert group_times(costs, world, 2) == [17, 10]

    def test_ranks_per_group_must_divide_the_world(self) -> None:
        # Arrange
        costs = [[1], [1], [1], [1]]

        # Act / Assert
        with pytest.raises(ValueError, match=r"ranks_per_group \(3\) must divide"):
            balanced_row_plan(costs, 3)


class TestBalanceRowsAcrossRanks:
    @requires_gloo
    def test_rows_move_with_all_their_fields(self) -> None:
        spawn_ranks(balance_worker)

    @requires_gloo
    def test_values_keep_their_dtype_shape_and_pad(self) -> None:
        spawn_ranks(values_worker)

    @requires_gloo
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            # Gloo receives on CPU, so CUDA rows mix devices.
            pytest.param(
                "cuda",
                marks=[
                    pytest.mark.gpu,
                    pytest.mark.skipif(
                        not torch.cuda.is_available(), reason="requires CUDA"
                    ),
                ],
            ),
        ],
    )
    def test_rows_stay_on_their_devices(self, device: str) -> None:
        spawn_ranks(devices_worker, device)

    @requires_gloo
    def test_fewer_rows_than_ranks_are_left_in_place(self) -> None:
        spawn_ranks(too_few_rows_worker)

    def test_a_single_process_is_left_in_place(self) -> None:
        # Arrange
        rows = rank_one_rows()
        advantages, episode_tokens = row_values(rows, 0)

        # Act
        balanced = balance_rows_across_ranks(
            rows, np.arange(1), [(advantages, 0.0), (episode_tokens, 0.0)], PAD, None
        )

        # Assert
        assert balanced is None
