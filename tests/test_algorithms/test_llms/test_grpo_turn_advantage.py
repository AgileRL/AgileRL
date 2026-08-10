# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for turn-masked GRPO advantage statistics.

Covers ``_calculate_turn_advantage`` with ``turn_mask`` and
``_turn_broadcast_advantages`` fall-back behaviour. Pure CPU: only the
advantage helpers are bound onto a stub; no DeepSpeed or vLLM.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl.algorithms.grpo import GRPO


class _AdvStub:
    """Stand-in carrying the state the turn-advantage helpers read."""

    def __init__(
        self,
        *,
        group_size: int = 4,
        adv_norm: str = "mean_std",
        turn_advantage_trajectory_fallback: bool = True,
        device: str = "cpu",
    ) -> None:
        self.group_size = group_size
        self.adv_norm = adv_norm
        self.turn_advantage_trajectory_fallback = turn_advantage_trajectory_fallback
        self.device = device

    _assert_batch_divisible_by_group = GRPO._assert_batch_divisible_by_group
    _calculate_advantage = GRPO._calculate_advantage
    _calculate_turn_advantage = GRPO._calculate_turn_advantage
    _turn_broadcast_advantages = GRPO._turn_broadcast_advantages


def _mask_stub(adv_norm: str = "mean_std", group_size: int = 4) -> _AdvStub:
    """Stub for the turn-mask advantage statistics (``group_size`` of 4)."""
    return _AdvStub(group_size=group_size, adv_norm=adv_norm)


def _worked_example_rewards() -> torch.Tensor:
    """Group of 4 where only the first sample stopped before the third turn."""
    return torch.tensor(
        [
            [0.05, 0.0, 0.0],
            [0.05, 0.05, 0.05],
            [0.05, 0.05, 0.05],
            [0.05, 0.05, 0.05],
        ],
    )


def _unmasked_turn_advantage(
    rewards: torch.Tensor,
    group_size: int,
    adv_norm: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-turn group-relative advantage over the full block, mask-free."""
    batch, num_turns = rewards.shape
    grouped = rewards.view(-1, group_size, num_turns)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if adv_norm == "mean_only":
        advantage = centered
    else:
        advantage = centered / (grouped.std(dim=1, keepdim=True) + eps)
    return advantage.reshape(batch, num_turns)


def _episode_tensors(
    lengths: list[int],
    num_turns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn ids and action masks for episodes of the given turn counts."""
    turn_ids = torch.full((len(lengths), num_turns), -1, dtype=torch.int64)
    for row, length in enumerate(lengths):
        turn_ids[row, :length] = torch.arange(length)
    return turn_ids, (turn_ids >= 0).to(torch.float32)


class TestGRPOMaskedTurnAdvantage:
    """``_calculate_turn_advantage`` with ``turn_mask``: per-turn group
    statistics over the members that actually played the turn.
    """

    def test_padding_fabricates_advantage_without_a_mask(self):
        rewards = _worked_example_rewards()

        unmasked = _mask_stub()._calculate_turn_advantage(rewards)

        assert torch.allclose(
            unmasked[:, 2],
            torch.tensor([-1.5, 0.5, 0.5, 0.5]),
            atol=1e-4,
        )

    def test_mask_removes_fabricated_advantage(self):
        rewards = _worked_example_rewards()
        turn_mask = torch.ones(4, 3, dtype=torch.bool)
        turn_mask[0, 2] = False

        masked = _mask_stub()._calculate_turn_advantage(rewards, turn_mask=turn_mask)

        assert torch.allclose(masked[:, 2], torch.zeros(4), atol=1e-6)
        assert torch.isfinite(masked).all()

    def test_a_genuinely_low_reward_keeps_its_negative_advantage(self):
        rewards = torch.tensor(
            [
                [0.05, -0.1],
                [0.05, 0.05],
                [0.05, 0.05],
                [0.05, 0.05],
            ],
        )
        turn_mask = torch.ones(4, 2, dtype=torch.bool)

        masked = _mask_stub()._calculate_turn_advantage(rewards, turn_mask=turn_mask)

        assert masked[0, 1].item() == pytest.approx(-1.5, abs=1e-3)
        assert masked[1, 1].item() == pytest.approx(0.5, abs=1e-3)

    def test_a_turn_nobody_played_is_exactly_zero(self):
        rewards = torch.tensor(
            [
                [0.05, 0.0],
                [0.05, 0.0],
                [0.05, 0.0],
                [0.05, 0.0],
            ],
        )
        turn_mask = torch.ones(4, 2, dtype=torch.bool)
        turn_mask[:, 1] = False

        masked = _mask_stub()._calculate_turn_advantage(rewards, turn_mask=turn_mask)

        assert torch.equal(masked, torch.zeros(4, 2))

    def test_a_single_valid_member_yields_zero(self):
        rewards = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
        turn_mask = torch.zeros(4, 1, dtype=torch.bool)
        turn_mask[0, 0] = True

        masked = _mask_stub()._calculate_turn_advantage(rewards, turn_mask=turn_mask)

        assert torch.equal(masked, torch.zeros(4, 1))

    def test_mean_only_centers_over_the_valid_members(self):
        stub = _mask_stub(adv_norm="mean_only")
        rewards = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [0.1, 0.1],
                [0.1, 0.1],
            ],
        )
        turn_mask = torch.ones(4, 2, dtype=torch.bool)
        turn_mask[0, 1] = False

        masked = stub._calculate_turn_advantage(rewards, turn_mask=turn_mask)

        assert masked[0, 0].item() == pytest.approx(-0.075, abs=1e-6)
        assert masked[1, 0].item() == pytest.approx(0.025, abs=1e-6)
        assert masked[0, 1].item() == 0.0

    def test_mean_std_uses_an_unbiased_denominator(self):
        rewards = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.1],
                [0.0, 0.1],
                [0.0, 0.3],
            ],
        )
        turn_mask = torch.ones(4, 2, dtype=torch.bool)
        turn_mask[0, 1] = False

        masked = _mask_stub()._calculate_turn_advantage(rewards, turn_mask=turn_mask)

        valid = torch.tensor([0.1, 0.1, 0.3])
        expected = (valid - valid.mean()) / valid.std()
        assert masked[0, 1].item() == 0.0
        assert torch.allclose(masked[1:, 1], expected, atol=1e-4)

    @pytest.mark.parametrize("adv_norm", ["mean_only", "mean_std"])
    def test_no_mask_matches_the_unmasked_formula(self, adv_norm):
        stub = _mask_stub(adv_norm=adv_norm)
        rewards = torch.randn(8, 5)

        patched = stub._calculate_turn_advantage(rewards)
        original = _unmasked_turn_advantage(rewards, stub.group_size, adv_norm)

        assert torch.equal(patched, original)


class TestGRPOTurnBroadcastMasking:
    """``_turn_broadcast_advantages`` derives the turn mask from ``turn_ids``
    and falls back to the trajectory advantage for baseline-free cells.
    """

    def test_unplayed_turns_do_not_enter_the_group_statistic(self):
        rewards = _worked_example_rewards()
        turn_ids = torch.tensor(
            [
                [0, 1, -1, -1],
                [0, 1, 2, -1],
                [0, 1, 2, -1],
                [0, 1, 2, -1],
            ],
        )
        action_masks = torch.ones(4, 4)

        advantages = _mask_stub()._turn_broadcast_advantages(
            rewards, turn_ids, action_masks, 4
        )

        assert advantages[0, 1].item() == pytest.approx(-1.5, abs=1e-3)
        assert advantages[1, 1].item() == pytest.approx(0.5, abs=1e-3)
        assert advantages[:, 0].abs().max().item() == 0.0
        assert torch.isfinite(advantages).all()

    def test_a_fully_padded_row_is_zero(self):
        rewards = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.05, 0.05],
                [0.05, 0.05, 0.05],
                [0.05, 0.0, 0.05],
            ],
        )
        turn_ids = torch.tensor(
            [
                [-1, -1, -1, -1],
                [0, 1, 2, -1],
                [0, 1, 2, -1],
                [0, 1, 2, -1],
            ],
        )
        action_masks = torch.ones(4, 4)

        advantages = _mask_stub()._turn_broadcast_advantages(
            rewards, turn_ids, action_masks, 4
        )

        assert torch.equal(advantages[0], torch.zeros(4))
        assert advantages[3, 1].item() == pytest.approx(-1.1547, abs=1e-3)

    def test_flat_rewards_are_reshaped_before_masking(self):
        rewards = _worked_example_rewards().reshape(-1)
        turn_ids = torch.tensor(
            [
                [0, 1, -1, -1],
                [0, 1, 2, -1],
                [0, 1, 2, -1],
                [0, 1, 2, -1],
            ],
        )
        action_masks = torch.ones(4, 4)

        advantages = _mask_stub()._turn_broadcast_advantages(
            rewards, turn_ids, action_masks, 4
        )

        assert advantages[0, 1].item() == pytest.approx(-1.5, abs=1e-3)

    def test_a_lone_survivor_gets_the_trajectory_advantage(self):
        stub = _mask_stub()
        rewards = torch.zeros(4, 20)
        rewards[:3, 0] = 0.2
        rewards[3, 1:] = 0.05
        turn_ids, action_masks = _episode_tensors([1, 1, 1, 20], 20)

        advantages = stub._turn_broadcast_advantages(rewards, turn_ids, action_masks, 4)
        trajectory = stub._calculate_advantage(rewards.sum(dim=1)).reshape(-1)

        assert advantages[3, 1:].abs().min().item() > 0.0
        assert torch.allclose(advantages[3, 1:], trajectory[3].expand(19), atol=1e-6)
        assert trajectory[3].item() == pytest.approx(1.5, abs=1e-4)
        assert advantages[3, 0].item() == pytest.approx(-1.5, abs=1e-4)
        assert advantages[0, 0].item() == pytest.approx(0.5, abs=1e-4)
        assert torch.isfinite(advantages).all()

    def test_the_fallback_is_negative_for_a_worse_total_return(self):
        rewards = torch.zeros(4, 5)
        rewards[:, 0] = 1.0
        rewards[3, 1:] = -0.1
        turn_ids, action_masks = _episode_tensors([1, 1, 1, 5], 5)

        advantages = _mask_stub()._turn_broadcast_advantages(
            rewards, turn_ids, action_masks, 4
        )

        assert advantages[3, 1:].max().item() < 0.0
        assert torch.allclose(advantages[3, 1:], torch.full((4,), -1.5), atol=1e-4)

    def test_unplayed_turns_stay_zero_under_the_fallback(self):
        rewards = torch.tensor(
            [
                [5.0, 0.0, 0.0],
                [0.05, 0.05, 0.05],
                [0.05, 0.05, 0.05],
                [0.05, 0.0, 0.05],
            ],
        )
        turn_ids = torch.tensor(
            [
                [-1, -1, -1],
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
            ],
        )
        action_masks = torch.ones(4, 3)

        advantages = _mask_stub()._turn_broadcast_advantages(
            rewards, turn_ids, action_masks, 4
        )

        assert torch.equal(advantages[0], torch.zeros(3))

    def test_multi_member_cells_keep_the_masked_per_turn_statistic(self):
        stub = _mask_stub()
        rewards = torch.tensor(
            [
                [0.30, 0.0, 0.0, 0.0, 0.0],
                [0.10, 0.0, 0.0, 0.0, 0.0],
                [0.20, 0.40, -0.10, 0.0, 0.0],
                [-0.05, 0.15, 0.25, 0.35, 0.45],
            ],
        )
        turn_ids, action_masks = _episode_tensors([1, 1, 3, 5], 5)
        masked_only = stub._calculate_turn_advantage(rewards, turn_mask=turn_ids >= 0)
        expected = masked_only.gather(1, turn_ids.clamp(min=0)) * action_masks

        advantages = stub._turn_broadcast_advantages(rewards, turn_ids, action_masks, 4)

        assert torch.equal(advantages[:, :3], expected[:, :3])
        assert torch.equal(expected[3, 3:], torch.zeros(2))
        assert advantages[3, 3:].abs().min().item() > 0.0

    def test_identical_total_returns_give_a_finite_zero_fallback(self):
        rewards = torch.zeros(4, 4)
        rewards[:3, 0] = 1.0
        rewards[3, :] = 0.25
        turn_ids, action_masks = _episode_tensors([1, 1, 1, 4], 4)

        advantages = _mask_stub()._turn_broadcast_advantages(
            rewards, turn_ids, action_masks, 4
        )

        assert torch.isfinite(advantages).all()
        assert torch.equal(advantages[3, 1:], torch.zeros(3))

    def test_the_fallback_uses_each_samples_own_group(self):
        stub = _mask_stub()
        rewards = torch.zeros(8, 5)
        rewards[:4, 0] = 0.1
        rewards[3, 1:] = 0.1
        rewards[4:7, 0] = 1.0
        rewards[7, 1:] = 0.05
        turn_ids, action_masks = _episode_tensors([1, 1, 1, 5, 1, 1, 1, 5], 5)

        advantages = stub._turn_broadcast_advantages(rewards, turn_ids, action_masks, 8)
        trajectory = stub._calculate_advantage(rewards.sum(dim=1)).reshape(-1)

        assert torch.allclose(advantages[3, 1:], torch.full((4,), 1.5), atol=1e-4)
        assert torch.allclose(advantages[7, 1:], torch.full((4,), -1.5), atol=1e-4)
        assert torch.allclose(advantages[3, 1:], trajectory[3].expand(4), atol=1e-6)
        assert torch.allclose(advantages[7, 1:], trajectory[7].expand(4), atol=1e-6)

    def test_disabling_the_fallback_leaves_baseline_free_cells_at_zero(self):
        stub = _AdvStub(
            group_size=4,
            adv_norm="mean_std",
            turn_advantage_trajectory_fallback=False,
        )
        rewards = torch.zeros(4, 20)
        rewards[:3, 0] = 0.2
        rewards[3, 1:] = 0.05
        turn_ids, action_masks = _episode_tensors([1, 1, 1, 20], 20)

        advantages = stub._turn_broadcast_advantages(rewards, turn_ids, action_masks, 4)

        assert torch.equal(advantages[3, 1:], torch.zeros(19))
        assert advantages[3, 0].item() == pytest.approx(-1.5, abs=1e-4)
