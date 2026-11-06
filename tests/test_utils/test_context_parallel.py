# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for context-parallel pad, shard, gather, and mixer all-to-all."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from agilerl.distributed.context_parallel import (
    all_to_all_hidden_to_seq,
    all_to_all_seq_to_hidden,
    gather_for_cp,
    pad_right_for_cp,
    shard_for_cp,
    validate_cp_degree,
    validate_cp_heads,
    validate_cp_seq_len,
    validate_cp_style,
)
from tests.rank_launch import run_ranks

_DIST_ENV = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


@pytest.fixture(autouse=True)
def _clean_dist_state():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    saved = {var: os.environ.pop(var, None) for var in _DIST_ENV}
    yield
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    for var, value in saved.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


class TestValidateCpDegree:
    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="cp must be >= 1"):
            validate_cp_degree(0)

    def test_rejects_bool(self) -> None:
        with pytest.raises(TypeError, match="cp must be an int"):
            validate_cp_degree(True)


class TestValidateCpStyle:
    def test_accepts_ulysses(self) -> None:
        assert validate_cp_style("ulysses") == "ulysses"

    def test_rejects_ring(self) -> None:
        with pytest.raises(ValueError, match="only 'ulysses'"):
            validate_cp_style("ring")


class TestValidateCpHeads:
    def test_inert_at_cp_one(self) -> None:
        validate_cp_heads(7, 3, 1)

    def test_query_heads_must_divide_cp(self) -> None:
        with pytest.raises(ValueError, match="num_attention_heads"):
            validate_cp_heads(7, 7, 2)

    def test_gqa_replicate_path_when_cp_divides_by_kv_heads(self) -> None:
        validate_cp_heads(8, 2, 4)

    def test_rejects_when_neither_kv_split_nor_replicate_works(self) -> None:
        with pytest.raises(ValueError, match="Ulysses GQA"):
            validate_cp_heads(8, 3, 2)


class TestValidateCpSeqLen:
    def test_inert_at_cp_one(self) -> None:
        validate_cp_seq_len(5, 1)

    def test_rejects_uneven_shards(self) -> None:
        with pytest.raises(ValueError, match="must be divisible by cp"):
            validate_cp_seq_len(5, 2)


class TestPadRightForCp:
    def test_inert_at_cp_one(self) -> None:
        ids = torch.arange(6).reshape(2, 3)

        padded_ids = pad_right_for_cp(ids, 1, pad_token_id=0)[0]

        assert padded_ids is ids

    def test_pads_action_frame_to_a_multiple_of_cp(self) -> None:
        ids = torch.ones(2, 6, dtype=torch.long)
        mask = torch.ones(2, 6, dtype=torch.long)
        labels = torch.full((2, 5), 7, dtype=torch.long)

        padded_ids, padded_mask, padded_labels = pad_right_for_cp(
            ids, 4, 0, mask, labels, pad_values=(0, -100)
        )

        assert padded_ids.shape == (2, 9)
        assert padded_mask.shape == (2, 9)
        assert padded_labels.shape == (2, 8)
        assert (padded_ids[:, 6:] == 0).all()
        assert (padded_mask[:, 6:] == 0).all()
        assert (padded_labels[:, 5:] == -100).all()
        assert (padded_ids.shape[1] - 1) % 4 == 0

    def test_pads_sequence_dim_of_per_position_support(self) -> None:
        ids = torch.ones(2, 6, dtype=torch.long)
        support = torch.full((2, 5, 3), 7, dtype=torch.long)

        _, padded_support = pad_right_for_cp(ids, 4, 0, support)

        assert padded_support.shape == (2, 8, 3)
        assert torch.equal(padded_support[:, :5], support)
        assert (padded_support[:, 5:] == 0).all()

    def test_skips_none_extras(self) -> None:
        ids = torch.ones(1, 3, dtype=torch.long)

        padded_ids, extra = pad_right_for_cp(ids, 2, 0, None)

        assert extra is None
        assert padded_ids.shape == (1, 3)

    def test_keeps_none_extras_when_padding(self) -> None:
        ids = torch.ones(1, 6, dtype=torch.long)
        labels = torch.full((1, 5), 7, dtype=torch.long)

        padded_ids, extra, padded_labels = pad_right_for_cp(
            ids, 4, 0, None, labels, pad_values=(0, -100)
        )

        assert extra is None
        assert padded_ids.shape == (1, 9)
        assert padded_labels.tolist() == [[7, 7, 7, 7, 7, -100, -100, -100]]


class TestAllToAllSeqToHidden:
    def test_inert_at_cp_one(self) -> None:
        data = torch.zeros(1, 4, 6)

        assert all_to_all_seq_to_hidden(data, 1, None) is data

    def test_rejects_hidden_not_divisible_by_cp(self) -> None:
        with pytest.raises(ValueError, match=r"hidden dim \(6\) must be divisible"):
            all_to_all_seq_to_hidden(torch.zeros(1, 4, 6), 4, None)


class TestAllToAllHiddenToSeq:
    def test_rejects_sequence_not_divisible_by_cp(self) -> None:
        with pytest.raises(
            ValueError, match=r"global sequence length \(6\) must be divisible"
        ):
            all_to_all_hidden_to_seq(torch.zeros(1, 6, 2), 4, None)


class TestShardForCp:
    def test_inert_at_cp_one(self) -> None:
        data = torch.arange(6).reshape(1, 6)

        assert shard_for_cp(data, 0, 1) is data

    def test_returns_this_ranks_even_slice(self) -> None:
        data = torch.arange(8).reshape(1, 8)

        assert shard_for_cp(data, 0, 2).tolist() == [[0, 1, 2, 3]]
        assert shard_for_cp(data, 1, 2).tolist() == [[4, 5, 6, 7]]

    def test_rejects_uneven_sequence(self) -> None:
        with pytest.raises(ValueError, match="divisible by cp size"):
            shard_for_cp(torch.arange(5).reshape(1, 5), 0, 2)


def _gather_and_mixer_roundtrip_worker(rank: int, world_size: int) -> None:
    group = dist.group.WORLD
    full = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
    shard = shard_for_cp(full, rank, world_size)
    gathered = gather_for_cp(shard, group)
    assert torch.equal(gathered, full)

    hidden = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    hidden = hidden + rank
    restored = all_to_all_hidden_to_seq(
        all_to_all_seq_to_hidden(hidden, world_size, group),
        world_size,
        group,
    )
    assert torch.allclose(restored, hidden)


@pytest.mark.multi_rank
class TestCpCollectives:
    def test_gather_and_mixer_all_to_all_roundtrip(self) -> None:
        run_ranks(_gather_and_mixer_roundtrip_worker)
