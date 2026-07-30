# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`agilerl.components.llm_rollout_buffer`."""

import random

import numpy as np
import pytest
import torch

from agilerl.components.llm_rollout_buffer import (
    LLMExperienceBatch,
    LLMRolloutBuffer,
    RolloutGroup,
    Trajectory,
)


def make_trajectory(seq_len, n_action=None, max_turns=2, with_logps=False):
    """One trajectory with ids ``(1, T)`` and mask/turn ids ``(1, T-1)``."""
    if n_action is None:
        n_action = max(1, (seq_len - 1) // 2)
    tokens = torch.arange(1, seq_len + 1, dtype=torch.long).reshape(1, seq_len)
    turn_ids = torch.full((1, seq_len - 1), -1, dtype=torch.long)
    turn_ids[0, :n_action] = 0
    rewards = torch.zeros(max_turns)
    rewards[0] = 1.0
    return Trajectory(
        completion_ids=tokens,
        action_masks=turn_ids >= 0,
        turn_ids=turn_ids,
        rewards=rewards,
        sampling_logps=torch.zeros(n_action) if with_logps else None,
    )


def make_group(group_size, lengths=None, max_turns=2, with_logps=False):
    lengths = lengths or [8] * group_size
    return RolloutGroup(
        group_size=group_size,
        trajectories=[
            make_trajectory(length, max_turns=max_turns, with_logps=with_logps)
            for length in lengths
        ],
    )


class TestTrajectory:
    @pytest.mark.parametrize("field", ["action_masks", "turn_ids"])
    def test_wrong_length_raises(self, field):
        kwargs = {
            "completion_ids": torch.ones(1, 8, dtype=torch.long),
            "action_masks": torch.ones(1, 7, dtype=torch.bool),
            "turn_ids": torch.zeros(1, 7, dtype=torch.long),
            "rewards": torch.ones(2),
        }
        kwargs[field] = kwargs[field][:, :-1]
        with pytest.raises(ValueError, match="completion_ids - 1"):
            Trajectory(**kwargs)

    def test_is_frozen(self):
        traj = make_trajectory(8)
        with pytest.raises(ValueError, match="frozen"):
            traj.rewards = torch.zeros(2)

    def test_tensors_held_by_reference(self):
        tokens = torch.arange(1, 9, dtype=torch.long).reshape(1, 8)
        turn_ids = torch.zeros(1, 7, dtype=torch.long)
        traj = Trajectory(
            completion_ids=tokens,
            action_masks=turn_ids >= 0,
            turn_ids=turn_ids,
            rewards=torch.ones(2),
        )
        assert traj.completion_ids is tokens
        assert traj.turn_ids is turn_ids


class TestRolloutGroup:
    def test_trajectory_count_must_match_group_size(self):
        with pytest.raises(ValueError, match="length group_size"):
            RolloutGroup(group_size=3, trajectories=[make_trajectory(8)])

    def test_group_size_must_be_positive(self):
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            RolloutGroup(group_size=0, trajectories=[])


class TestBuffer:
    def test_invalid_memory_size_raises(self):
        with pytest.raises(ValueError, match="memory_size"):
            LLMRolloutBuffer(memory_size=0)

    def test_empty_buffer_pops_empty_batch(self):
        buf = LLMRolloutBuffer(memory_size=2)
        assert len(buf) == 0
        assert buf.memory_size == 2
        batch = buf.pop_all()
        assert isinstance(batch, LLMExperienceBatch)
        assert batch.is_empty
        assert batch.turn_ids is None
        assert len(batch) == 0

    def test_add_and_drain(self):
        buf = LLMRolloutBuffer(memory_size=4)
        assert buf.add_group(make_group(2)) == 0
        assert buf.add_group(make_group(2)) == 0
        assert len(buf) == 2
        batch = buf.pop_all()
        assert len(batch) == 4
        assert len(buf) == 0

    def test_oldest_group_evicted_at_capacity(self):
        buf = LLMRolloutBuffer(memory_size=2)
        for length in (6, 8, 10):
            buf.add_group(make_group(1, lengths=[length]))
        assert len(buf) == 2
        batch = buf.pop_all()
        assert batch.completion_lengths.tolist() == [8, 10]

    def test_group_size_change_clears_buffer(self):
        buf = LLMRolloutBuffer(memory_size=8)
        buf.add_group(make_group(2))
        buf.add_group(make_group(2))
        assert buf.add_group(make_group(3)) == 2
        assert len(buf) == 1
        assert len(buf.pop_all()) == 3

    def test_pop_groups_consumes_oldest(self):
        buf = LLMRolloutBuffer(memory_size=4)
        for length in (6, 8, 10):
            buf.add_group(make_group(1, lengths=[length]))
        batch = buf.pop_groups(2)
        assert batch is not None
        assert batch.completion_lengths.tolist() == [6, 8]
        assert len(buf) == 1

    def test_pop_groups_returns_none_when_short(self):
        buf = LLMRolloutBuffer(memory_size=4)
        buf.add_group(make_group(1))
        assert buf.pop_groups(2) is None
        assert len(buf) == 1

    def test_clear(self):
        buf = LLMRolloutBuffer(memory_size=4)
        buf.add_group(make_group(2))
        buf.clear()
        assert len(buf) == 0
        assert buf.add_group(make_group(3)) == 0


class TestCollate:
    def test_ragged_tensors_passed_through_by_reference(self):
        group = make_group(2, lengths=[6, 9])
        buf = LLMRolloutBuffer(memory_size=1)
        buf.add_group(group)
        batch = buf.pop_all()
        for i, traj in enumerate(group.trajectories):
            assert batch.completion_ids[i] is traj.completion_ids
            assert batch.action_masks[i] is traj.action_masks

    def test_turn_ids_padded_with_minus_one(self):
        buf = LLMRolloutBuffer(memory_size=1)
        buf.add_group(make_group(2, lengths=[6, 9]))
        batch = buf.pop_all()
        assert batch.turn_ids.shape == (2, 8)
        assert torch.all(batch.turn_ids[0, 5:] == -1)

    def test_rewards_stacked_and_float(self):
        buf = LLMRolloutBuffer(memory_size=2)
        buf.add_group(make_group(2, max_turns=3))
        batch = buf.pop_all()
        assert batch.rewards.shape == (2, 3)
        assert batch.rewards.dtype == torch.float32
        assert torch.equal(batch.rewards[:, 0], torch.ones(2))

    def test_rewards_padded_across_differing_widths(self):
        buf = LLMRolloutBuffer(memory_size=2)
        buf.add_group(make_group(1, max_turns=2))
        buf.add_group(make_group(1, max_turns=4))
        batch = buf.pop_all()
        assert batch.rewards.shape == (2, 4)
        assert torch.all(batch.rewards[0, 2:] == 0.0)

    def test_completion_lengths(self):
        buf = LLMRolloutBuffer(memory_size=1)
        buf.add_group(make_group(3, lengths=[5, 7, 11]))
        assert buf.pop_all().completion_lengths.tolist() == [5, 7, 11]

    def test_sampling_logps_none_when_absent(self):
        buf = LLMRolloutBuffer(memory_size=1)
        buf.add_group(make_group(2))
        assert buf.pop_all().sampling_logps is None

    def test_sampling_logps_kept_when_present(self):
        buf = LLMRolloutBuffer(memory_size=1)
        buf.add_group(make_group(2, with_logps=True))
        logps = buf.pop_all().sampling_logps
        assert logps is not None
        assert all(lp is not None for lp in logps)

    def test_experiences_tuple(self):
        buf = LLMRolloutBuffer(memory_size=1)
        buf.add_group(make_group(2))
        batch = buf.pop_all()
        completion_ids, action_masks, rewards = batch.experiences()
        assert completion_ids is batch.completion_ids
        assert action_masks is batch.action_masks
        assert rewards is batch.rewards

    def test_groups_stay_contiguous_and_ordered(self):
        buf = LLMRolloutBuffer(memory_size=2)
        buf.add_group(make_group(2, lengths=[4, 5]))
        buf.add_group(make_group(2, lengths=[6, 7]))
        assert buf.pop_all().completion_lengths.tolist() == [4, 5, 6, 7]


class TestSyncTrainerWiring:
    @staticmethod
    def _make_rollout(batch_size, group_size, max_turns, rng):
        cids, masks, turns, rewards = [], [], [], []
        for _ in range(batch_size * group_size):
            traj = make_trajectory(rng.randint(3, 20), max_turns=max_turns)
            cids.append(traj.completion_ids)
            masks.append(traj.action_masks)
            turns.append(traj.turn_ids)
            rewards.append(traj.rewards)
        return cids, masks, turns, rewards

    def test_matches_hand_rolled_assembly(self):
        from agilerl.rollouts.on_policy import buffer_llm_rollouts
        from agilerl.utils.algo_utils import stack_and_pad_experiences

        rng = random.Random(0)
        group_size, batch_size, max_turns = 4, 3, 2
        cids, masks, turns, rewards = self._make_rollout(
            batch_size, group_size, max_turns, rng
        )

        normalized = [r.unsqueeze(0) if r.dim() == 1 else r for r in rewards]
        (turn_ids_padded,) = stack_and_pad_experiences(turns, padding_values=[-1])
        (rewards_2d,) = stack_and_pad_experiences(normalized, padding_values=[0.0])
        old_c, old_m = stack_and_pad_experiences(cids, masks, padding_values=[0, False])

        batch = buffer_llm_rollouts(cids, masks, turns, rewards, group_size=group_size)
        new_c, new_m = stack_and_pad_experiences(
            batch.completion_ids, batch.action_masks, padding_values=[0, False]
        )

        assert torch.equal(new_c, old_c)
        assert torch.equal(new_m, old_m)
        assert torch.equal(batch.rewards, rewards_2d.float())
        assert torch.equal(batch.turn_ids, turn_ids_padded)
        assert batch.completion_lengths.float().mean().item() == pytest.approx(
            np.mean([c.shape[1] for c in cids])
        )

    def test_rollout_tensors_are_not_copied(self):
        from agilerl.rollouts.on_policy import buffer_llm_rollouts

        cids, masks, turns, rewards = self._make_rollout(2, 2, 2, random.Random(3))
        batch = buffer_llm_rollouts(cids, masks, turns, rewards, group_size=2)
        assert all(a is b for a, b in zip(batch.completion_ids, cids, strict=True))
        assert all(a is b for a, b in zip(batch.action_masks, masks, strict=True))

    def test_non_divisible_raises(self):
        from agilerl.rollouts.on_policy import buffer_llm_rollouts

        cids, masks, turns, rewards = self._make_rollout(1, 3, 2, random.Random(1))
        with pytest.raises(ValueError, match="divisible"):
            buffer_llm_rollouts(cids, masks, turns, rewards, group_size=2)

    def test_empty_rollout(self):
        from agilerl.rollouts.on_policy import buffer_llm_rollouts

        assert buffer_llm_rollouts([], [], [], [], group_size=4).is_empty

    def test_mismatched_mask_length_fails_loudly(self):
        from agilerl.rollouts.on_policy import buffer_llm_rollouts

        cids = [torch.ones(1, 8, dtype=torch.long)]
        masks = [torch.ones(1, 8, dtype=torch.bool)]
        turns = [torch.zeros(1, 8, dtype=torch.long)]
        rewards = [torch.ones(2)]
        with pytest.raises(ValueError, match="completion_ids - 1"):
            buffer_llm_rollouts(cids, masks, turns, rewards, group_size=1)
