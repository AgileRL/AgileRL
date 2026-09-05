# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GRPO group-relative advantage computation."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import torch

from agilerl.algorithms.core.advantage_granularity import (
    resolve_batch_advantage_granularity,
)
from agilerl.utils.llm_utils import (
    baseline_free_turn_cells,
    masked_whiten,
)


class GRPOAdvantageMixin:
    """Group-relative advantages for GRPO / GSPO / CISPO."""

    def _setup_advantage_options(
        self,
        adv_norm: str,
        group_size: int,
        advantage_granularity: str,
        action_granularity: str | None,
        whiten_advantages: bool,
        adv_clip_range: float | None,
        filter_zero_adv: bool,
        adv_filter_eps: float,
        turn_advantage_trajectory_fallback: bool,
    ) -> None:
        """Validate and store the advantage-computation options."""
        if adv_norm not in {"mean_std", "mean_only"}:
            msg = (
                f"Invalid adv_norm '{adv_norm}'. Expected one of "
                "['mean_std', 'mean_only']."
            )
            raise ValueError(msg)
        if group_size < 2:
            msg = (
                f"group_size must be >= 2 for GRPO-style group-relative "
                f"advantages; got {group_size}. A group of one yields a zero "
                "advantage for every sample (reward minus its own mean), so the "
                "policy receives no gradient signal."
            )
            raise ValueError(msg)
        if adv_clip_range is not None and adv_clip_range <= 0:
            msg = "adv_clip_range must be > 0 when provided."
            raise ValueError(msg)
        if adv_filter_eps < 0:
            msg = "adv_filter_eps must be >= 0."
            raise ValueError(msg)
        if action_granularity is not None:
            warnings.warn(
                "action_granularity is deprecated; use advantage_granularity.",
                DeprecationWarning,
                stacklevel=3,
            )
            advantage_granularity = action_granularity
        if advantage_granularity not in {"auto", "trajectory", "turn"}:
            msg = (
                f"Invalid advantage_granularity '{advantage_granularity}'. Expected "
                "one of ['auto', 'trajectory', 'turn']. The GRPO family has no "
                "token-level advantage (group-relative needs a reward per unit, "
                "and tokens have none)."
            )
            raise ValueError(msg)
        self.adv_norm = adv_norm
        self.group_size = group_size
        self.advantage_granularity = advantage_granularity
        self.whiten_advantages = whiten_advantages
        self.adv_clip_range = adv_clip_range
        self.filter_zero_adv = filter_zero_adv
        self.adv_filter_eps = adv_filter_eps
        self.turn_advantage_trajectory_fallback = turn_advantage_trajectory_fallback
    def _calculate_advantages(
        self,
        rewards: torch.Tensor,
        token_ids: torch.Tensor,
        action_masks: torch.Tensor,
        turn_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, npt.NDArray]:
        """Group-relative advantages at the resolved granularity, post-processed.

        Post-processing (zero-filter / whiten / clip) is shape-agnostic across
        per-trajectory ``(B, 1)`` and per-turn-broadcast ``(B, T-1)``
        advantages. Returns the advantages and the indices of samples that
        survive the zero-advantage filter (all samples when it is disabled).
        Multi-process runs apply the filter by zeroing the advantages of
        filtered samples and returning every index: each rank keeps its full
        batch, so all ranks run the same forward/backward collective schedule
        regardless of how many samples each one filters.
        ``"turn"`` granularity requires ``turn_ids``.
        """
        num_samples = token_ids.shape[0]
        resolved = self._resolve_advantage_granularity(turn_ids, rewards)
        if (
            self.advantage_granularity == "auto"
            and resolved == "turn"
            and self.importance_sampling_level == "trajectory"
        ):
            warnings.warn(
                "advantage_granularity='turn' with "
                "importance_sampling_level='trajectory' applies one "
                "completion-level ratio to per-turn advantages. Set "
                "advantage_granularity='trajectory' to match, or use token/turn "
                "importance sampling to keep turn advantages.",
                stacklevel=2,
            )
        if resolved == "turn":
            if turn_ids is None:
                msg = "advantage_granularity='turn' requires turn_ids; got None."
                raise ValueError(msg)
            advantages = self._turn_broadcast_advantages(
                rewards, turn_ids, action_masks, num_samples
            )
        else:
            advantages = self._trajectory_advantages(rewards, num_samples, token_ids)

        active_adv_mask = None
        if self.filter_zero_adv:
            per_sample_abs = (
                advantages.detach().reshape(num_samples, -1).abs().amax(dim=-1)
            )
            active_adv_mask = per_sample_abs > self.adv_filter_eps
        if self.whiten_advantages:
            advantages = self._whiten_advantages(
                advantages, action_masks, active_adv_mask
            )
        if self.adv_clip_range is not None:
            advantages = advantages.clamp(-self.adv_clip_range, self.adv_clip_range)

        if active_adv_mask is None:
            return advantages, np.arange(num_samples)
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            advantages = advantages * active_adv_mask.unsqueeze(-1).to(advantages.dtype)
            return advantages, np.arange(num_samples)
        return advantages, np.where(active_adv_mask.detach().cpu().numpy())[0]
    def _assert_batch_divisible_by_group(self, num_samples: int) -> None:
        """Require the trajectory batch to split evenly into GRPO groups.

        Called from :meth:`learn` *after* rewards-cardinality validation so a
        rewards/trajectory count mismatch surfaces its own error first.

        :param num_samples: Number of trajectories in the batch.
        :type num_samples: int
        :raises ValueError: If ``num_samples`` is not divisible by
            ``group_size``.
        """
        if num_samples % self.group_size != 0:
            msg = (
                f"Batch size ({num_samples}) must be divisible by "
                f"group_size ({self.group_size}) for GRPO."
            )
            raise ValueError(msg)
    def _calculate_advantage(
        self,
        rewards: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Calculate the group relative advantage for each groups reward.

        :param rewards: Tensor of rewards.
        :type rewards: torch.Tensor
        :param eps: Epsilon to prevent zero division error, defaults to 1e-8
        :type eps: float, optional
        :return: Tensor of group relative advantages.
        :rtype: torch.Tensor
        :raises ValueError: If the number of elements in ``rewards`` is not
            divisible by ``group_size``.
        """
        numel = rewards.numel()
        if numel % self.group_size != 0:
            msg = (
                f"Rewards must have a total element count divisible by "
                f"group_size ({self.group_size}); got {numel} elements."
            )
            raise ValueError(msg)
        rewards = rewards.view(-1, self.group_size)
        centered_rewards = rewards - rewards.mean(dim=1, keepdim=True)
        if self.adv_norm == "mean_only":
            advantage = centered_rewards
        else:
            advantage = centered_rewards / (rewards.std(dim=1, keepdim=True) + eps)
        return advantage.flatten().unsqueeze(1)
    def _calculate_turn_advantage(
        self,
        rewards: torch.Tensor,
        eps: float = 1e-8,
        turn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Group-relative advantage computed independently per turn.

        Treats each turn as a separate RL action: for turn ``k`` the reward is
        normalized within its group of ``group_size`` completions (the same
        group-relative scheme as :meth:`_calculate_advantage`, applied per
        turn). The caller then broadcasts each ``(sample, turn)`` advantage to
        every token of that turn via ``turn_ids``. ``turn_mask`` restricts each
        group statistic to the members that played the turn, so an episode that
        ended early contributes nothing to the turns it never reached, and cells
        left without a baseline come back as zero.

        :param rewards: Per-turn rewards ``(batch, max_turns)``; the batch dim
            is grouped in contiguous blocks of ``group_size``.
        :type rewards: torch.Tensor
        :param eps: Epsilon guarding the per-turn std division.
        :type eps: float, optional
        :param turn_mask: Boolean ``(batch, max_turns)`` mask of played turns;
            ``None`` lets every entry participate in the statistics.
        :type turn_mask: torch.Tensor | None, optional
        :return: Per-turn advantages ``(batch, max_turns)``.
        :rtype: torch.Tensor
        :raises ValueError: If the batch size is not divisible by ``group_size``.
        """
        batch = rewards.shape[0]
        if batch % self.group_size != 0:
            msg = (
                f"Per-turn rewards batch ({batch}) must be divisible by "
                f"group_size ({self.group_size})."
            )
            raise ValueError(msg)
        num_turns = rewards.shape[1]
        grouped = rewards.view(-1, self.group_size, num_turns)

        if turn_mask is None:
            centered = grouped - grouped.mean(dim=1, keepdim=True)
            if self.adv_norm == "mean_only":
                advantage = centered
            else:
                advantage = centered / (grouped.std(dim=1, keepdim=True) + eps)
            return advantage.reshape(batch, num_turns)

        valid = turn_mask.reshape(-1, self.group_size, num_turns).to(grouped.dtype)
        count = valid.sum(dim=1, keepdim=True)
        mean = (grouped * valid).sum(dim=1, keepdim=True) / count.clamp(min=1.0)
        centered = (grouped - mean) * valid
        if self.adv_norm == "mean_only":
            advantage = centered
        else:
            denom = (count - 1.0).clamp(min=1.0)
            std = (centered.pow(2).sum(dim=1, keepdim=True) / denom).sqrt()
            advantage = centered / (std + eps)
        advantage = torch.where(count > 1, advantage, torch.zeros_like(advantage))
        return advantage.reshape(batch, num_turns)
    def _turn_broadcast_advantages(
        self,
        rewards: torch.Tensor,
        turn_ids: torch.Tensor,
        action_masks: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Per-turn group-relative advantages, broadcast to token positions.

        A turn is the action unit: each turn's reward is normalized within the
        group members that played it (:meth:`_calculate_turn_advantage`), then
        assigned to every token of that turn via ``turn_ids`` and masked to
        action positions. Under
        :attr:`turn_advantage_trajectory_fallback`, a played cell whose group
        has no second member at that turn takes the sample's trajectory
        advantage (:meth:`_calculate_advantage`) rather than zero.

        :param rewards: Per-turn rewards ``(B, max_turns)`` (or flat, reshaped).
        :type rewards: torch.Tensor
        :param turn_ids: ``(B, T-1)`` per-token turn indices (``-1`` = padding).
        :type turn_ids: torch.Tensor
        :param action_masks: ``(B, T-1)`` action-token mask.
        :type action_masks: torch.Tensor
        :param num_samples: Trajectory count in the batch.
        :type num_samples: int
        :return: ``(B, T-1)`` per-token advantages.
        :rtype: torch.Tensor
        :raises ValueError: If ``turn_ids`` reference more turns than rewards
            provide, or the batch is not divisible by ``group_size``.
        """
        self._assert_batch_divisible_by_group(num_samples)
        turn_rewards = (
            rewards if rewards.dim() > 1 else rewards.reshape(num_samples, -1)
        )
        safe_turn_ids = turn_ids.clamp(min=0).to(torch.int64)
        num_reward_turns = turn_rewards.shape[1]
        if int(safe_turn_ids.max().item()) >= num_reward_turns:
            msg = (
                "turn_ids reference a turn index beyond the number of "
                f"reward turns ({num_reward_turns}); rewards and "
                "turn_ids are misaligned."
            )
            raise ValueError(msg)
        turn_counts = torch.zeros(
            (turn_rewards.shape[0], num_reward_turns),
            dtype=torch.int64,
            device=safe_turn_ids.device,
        )
        # Clamping maps padding onto turn 0, so occupancy must accumulate: a
        # plain scatter_ has no defined winner among duplicate indices in a row.
        turn_counts.scatter_add_(1, safe_turn_ids, (turn_ids >= 0).to(torch.int64))
        turn_mask = (turn_counts > 0).to(turn_rewards.device)
        turn_advantages = self._calculate_turn_advantage(
            turn_rewards,
            turn_mask=turn_mask,
        ).to(self.device)
        if self.turn_advantage_trajectory_fallback:
            sparse = baseline_free_turn_cells(turn_mask, self.group_size).to(
                turn_advantages.device,
            )
            trajectory = (
                self._calculate_advantage(turn_rewards.sum(dim=1))
                .reshape(-1, 1)
                .to(turn_advantages.device)
                .expand_as(turn_advantages)
            )
            turn_advantages = torch.where(sparse, trajectory, turn_advantages)
        advantages = turn_advantages.gather(1, safe_turn_ids)  # (B, T-1)
        return advantages * action_masks.to(advantages.dtype)
    def _trajectory_advantages(
        self,
        rewards: torch.Tensor,
        num_samples: int,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Per-trajectory group-relative advantage ``(B, 1)``.

        Per-turn reward matrices ``(B, max_turns)`` are collapsed to episode
        returns before the group-relative normalization
        (:meth:`_calculate_advantage`).

        :param rewards: Per-trajectory or per-turn rewards.
        :type rewards: torch.Tensor
        :param num_samples: Trajectory count in the batch.
        :type num_samples: int
        :param token_ids: Completion ids, used only for error reporting.
        :type token_ids: torch.Tensor
        :return: ``(B, 1)`` per-trajectory advantages.
        :rtype: torch.Tensor
        :raises ValueError: If rewards don't collapse to one scalar per
            trajectory, or the batch is not divisible by ``group_size``.
        """
        if rewards.dim() > 1 and rewards.shape[0] == num_samples:
            rewards = rewards.sum(dim=1)
        rewards = rewards.flatten()
        if rewards.shape[0] != num_samples:
            msg = (
                "Rewards must provide one scalar per trajectory after "
                f"collapse: got rewards={tuple(rewards.shape)} and "
                f"token_ids={tuple(token_ids.shape)}."
            )
            raise ValueError(msg)
        self._assert_batch_divisible_by_group(num_samples)
        return self._calculate_advantage(rewards).to(self.device)
    def _resolve_advantage_granularity(
        self,
        turn_ids: torch.Tensor | None = None,
        rewards: torch.Tensor | None = None,
    ) -> str:
        """Return the unit at which group-relative advantages are computed.

        Independent of :attr:`importance_sampling_level` (the IS / ratio-pooling
        axis). ``"trajectory"`` and ``"turn"`` can pair with any IS level.
        ``"auto"`` is ``"turn"`` when the batch has per-turn rewards and any
        sample has more than one turn; otherwise ``"trajectory"``.

        :param turn_ids: Per-token turn index, or ``None``.
        :type turn_ids: torch.Tensor | None
        :param rewards: Trajectory scalars or per-turn rewards.
        :type rewards: torch.Tensor | None
        :return: ``"trajectory"`` or ``"turn"``.
        :rtype: str
        """
        has_turn_rewards = (
            rewards is not None and rewards.ndim > 1 and rewards.shape[-1] > 1
        )
        return resolve_batch_advantage_granularity(
            self.advantage_granularity,
            turn_ids,
            single_turn="trajectory",
            multi_turn="turn",
            can_use_multi_turn=has_turn_rewards,
        )
    def _whiten_advantages(
        self,
        advantages: torch.Tensor,
        action_masks: torch.Tensor,
        active_adv_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Whiten advantages, handling per-trajectory and per-token shapes.

        * Per-trajectory ``(B, 1)``: whiten across (active) samples — the
          original GRPO behavior.
        * Per-token / per-turn ``(B, T-1)``: whiten over valid action
          positions (optionally restricted to active samples).

        :param advantages: ``(B, 1)`` or ``(B, T-1)`` advantages.
        :type advantages: torch.Tensor
        :param action_masks: ``(B, T-1)`` action-token mask.
        :type action_masks: torch.Tensor
        :param active_adv_mask: Optional ``(B,)`` per-sample keep mask.
        :type active_adv_mask: torch.Tensor | None
        :return: Whitened advantages with the same shape as ``advantages``.
        :rtype: torch.Tensor
        """
        if advantages.dim() <= 1 or advantages.shape[-1] == 1:
            adv = advantages.reshape(-1)
            mask = (
                active_adv_mask
                if active_adv_mask is not None and active_adv_mask.any()
                else torch.ones_like(adv, dtype=torch.bool)
            )
        else:
            adv = advantages
            mask = action_masks.bool()
            if active_adv_mask is not None:
                mask = mask & active_adv_mask.unsqueeze(-1)
        if mask.sum() <= 1:
            # Fewer than two whitenable values: variance is undefined, leave
            # the advantages untouched rather than dividing by ~0.
            return advantages
        whitened = masked_whiten(adv, mask.to(adv.dtype), shift_mean=True)
        result = torch.where(mask, whitened, adv)
        return result.reshape(advantages.shape)
    def _apply_kl_advantage_shaping(
        self,
        advantages: torch.Tensor,
        kl: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ART-style zero-mean KL shaping to token advantages."""
        if not self.use_kl_advantage_shaping:
            return advantages
        mask_f = mask.float()
        masked_kl = kl * mask_f
        avg_kl = masked_kl.sum(dim=-1, keepdim=True) / mask_f.sum(
            dim=-1,
            keepdim=True,
        ).clamp(min=1.0)
        return advantages + self.beta * (avg_kl - masked_kl)
