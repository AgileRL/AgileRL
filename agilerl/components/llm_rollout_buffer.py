# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Bounded FIFO buffer of LLM rollouts, admitted and evicted one prompt group at a time.

The unit of storage is the **prompt group**: all ``group_size`` completions of one
prompt (one group per prompt for GRPO, ``group_size=1`` for PPO and REINFORCE, where
each group is a single trajectory). Groups are added and evicted whole, so the
group-divisibility the algorithms rely on holds by construction.

Trajectory tensors are held by reference and are never copied on the way in or out;
padding into rectangular batches happens once, at collate time. Synchronous training
uses :meth:`LLMRolloutBuffer.add_group` and :meth:`LLMRolloutBuffer.pop_all`.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agilerl.utils.algo_utils import stack_and_pad_experiences


class Trajectory(BaseModel):
    """One completed LLM trajectory.

    :param completion_ids: ``(1, T)`` token ids for the whole episode.
    :param action_masks: ``(1, T - 1)`` mask marking action (model-generated) positions.
    :param turn_ids: ``(1, T - 1)`` turn index per action token, ``-1`` elsewhere.
    :param rewards: ``(max_turns,)`` or ``(1, max_turns)`` per-turn rewards.
    :param sampling_logps: Optional 1-D generated-token sampling logprobs.
    """

    completion_ids: torch.Tensor
    action_masks: torch.Tensor
    turn_ids: torch.Tensor
    rewards: torch.Tensor
    sampling_logps: torch.Tensor | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_shapes(self) -> Trajectory:
        """Require the per-action-token fields to be one shorter than the token ids."""
        expected = int(self.completion_ids.shape[-1]) - 1
        for name, tensor in (
            ("action_masks", self.action_masks),
            ("turn_ids", self.turn_ids),
        ):
            if int(tensor.shape[-1]) != expected:
                msg = (
                    f"{name} must have length completion_ids - 1 ({expected}), "
                    f"got {int(tensor.shape[-1])}."
                )
                raise ValueError(msg)
        return self


class RolloutGroup(BaseModel):
    """The set of completions sampled together for one prompt.

    :param group_size: Number of completions sampled for the prompt.
    :param trajectories: Exactly ``group_size`` trajectories.
    """

    group_size: int = Field(ge=1)
    trajectories: list[Trajectory]

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_group_shape(self) -> RolloutGroup:
        """Require exactly ``group_size`` trajectories."""
        if len(self.trajectories) != self.group_size:
            msg = (
                f"trajectories must be a list of length group_size "
                f"({self.group_size}), got {len(self.trajectories)}."
            )
            raise ValueError(msg)
        return self


@dataclass(frozen=True)
class LLMExperienceBatch:
    """Collated batch drained from :class:`LLMRolloutBuffer`.

    ``completion_ids`` and ``action_masks`` are the ragged per-row tensors that
    ``learn()`` pads itself; ``rewards`` and ``turn_ids`` are pre-stacked rectangles.

    :param completion_ids: One ``(1, T)`` tensor per trajectory.
    :param action_masks: One ``(1, T - 1)`` tensor per trajectory.
    :param rewards: ``(B, max_turns)`` float tensor of per-turn rewards.
    :param turn_ids: ``(B, T_max - 1)`` tensor padded with ``-1``, or ``None`` when empty.
    :param completion_lengths: ``(B,)`` long tensor of per-row token counts.
    :param sampling_logps: Per-row logprob tensors, or ``None`` when none were captured.
    """

    completion_ids: list[torch.Tensor]
    action_masks: list[torch.Tensor]
    rewards: torch.Tensor
    turn_ids: torch.Tensor | None
    completion_lengths: torch.Tensor
    sampling_logps: list[torch.Tensor | None] | None = None

    def __len__(self) -> int:
        return len(self.completion_ids)

    @property
    def is_empty(self) -> bool:
        """Whether the batch holds no trajectories."""
        return len(self.completion_ids) == 0

    def experiences(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Return the ``(completion_ids, action_masks, rewards)`` learn tuple."""
        return (self.completion_ids, self.action_masks, self.rewards)


class LLMRolloutBuffer:
    """Bounded FIFO of rollout groups, evicting the oldest once ``memory_size`` is reached.

    :param memory_size: Maximum number of groups held.
    :type memory_size: int
    """

    def __init__(self, memory_size: int) -> None:
        if memory_size < 1:
            msg = f"memory_size must be >= 1, got {memory_size}"
            raise ValueError(msg)
        self._memory_size = int(memory_size)
        self._groups: deque[RolloutGroup] = deque(maxlen=self._memory_size)
        self._last_group_size: int | None = None
        self._lock = threading.Lock()

    @property
    def memory_size(self) -> int:
        """Maximum number of groups held."""
        return self._memory_size

    def __len__(self) -> int:
        with self._lock:
            return len(self._groups)

    def add_group(self, group: RolloutGroup) -> int:
        """Append one group, clearing the buffer first if ``group_size`` changed.

        :param group: The group to admit.
        :type group: RolloutGroup
        :return: Number of stale groups dropped because ``group_size`` changed.
        :rtype: int
        """
        with self._lock:
            return self._append_group(group)

    def pop_groups(self, n_groups: int) -> LLMExperienceBatch | None:
        """Consume the oldest ``n_groups`` groups, or return ``None`` if too few are held.

        :param n_groups: Number of groups to consume.
        :type n_groups: int
        :return: The collated batch, or ``None``.
        :rtype: LLMExperienceBatch | None
        """
        with self._lock:
            if len(self._groups) < n_groups:
                return None
            popped = [self._groups.popleft() for _ in range(n_groups)]
            return self.collate(popped)

    def pop_all(self) -> LLMExperienceBatch:
        """Drain every held group as one batch."""
        with self._lock:
            popped = list(self._groups)
            self._groups.clear()
            return self.collate(popped)

    def clear(self) -> None:
        """Drop all held groups."""
        with self._lock:
            self._groups.clear()
            self._last_group_size = None

    @staticmethod
    def collate(groups: Sequence[RolloutGroup]) -> LLMExperienceBatch:
        """Flatten groups into one batch, padding turn ids and rewards into rectangles.

        :param groups: Groups to collate, in order.
        :type groups: Sequence[RolloutGroup]
        :return: The collated batch.
        :rtype: LLMExperienceBatch
        """
        trajectories = [traj for group in groups for traj in group.trajectories]
        if not trajectories:
            return LLMExperienceBatch(
                completion_ids=[],
                action_masks=[],
                rewards=torch.zeros(0, 0),
                turn_ids=None,
                completion_lengths=torch.zeros(0, dtype=torch.long),
            )

        (turn_ids,) = stack_and_pad_experiences(
            [traj.turn_ids for traj in trajectories], padding_values=[-1]
        )
        (rewards,) = stack_and_pad_experiences(
            [
                traj.rewards.unsqueeze(0) if traj.rewards.dim() == 1 else traj.rewards
                for traj in trajectories
            ],
            padding_values=[0.0],
        )
        logps = [traj.sampling_logps for traj in trajectories]
        return LLMExperienceBatch(
            completion_ids=[traj.completion_ids for traj in trajectories],
            action_masks=[traj.action_masks for traj in trajectories],
            rewards=rewards.float(),
            turn_ids=turn_ids,
            completion_lengths=torch.tensor(
                [int(traj.completion_ids.shape[-1]) for traj in trajectories],
                dtype=torch.long,
            ),
            sampling_logps=logps if any(lp is not None for lp in logps) else None,
        )

    def _append_group(self, group: RolloutGroup) -> int:
        dropped = 0
        if (
            self._last_group_size is not None
            and group.group_size != self._last_group_size
        ):
            dropped = len(self._groups)
            self._groups.clear()
        self._last_group_size = group.group_size
        self._groups.append(group)
        return dropped
