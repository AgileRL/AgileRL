# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Pack finished LLM rollouts into a training batch.

Trajectories arrive grouped by prompt -- all ``group_size`` completions of one
prompt together -- and are padded into rectangles once, on the way into
:class:`LLMExperienceBatch`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agilerl.utils.algo_utils import stack_and_pad_experiences


@dataclass(frozen=True)
class EpisodeSegments:
    """Layout of an episode whose context restarted, stored as consecutive segments.

    Each segment is its own self-contained sequence. In the episode row they sit
    back to back; the action mask is ``False`` at the position that predicts a
    segment's first token.

    :param token_lengths: ``(S,)`` long tensor of segment token counts, ``S >= 2``.
    :param pixel_rows: ``(S,)`` long tensor of each segment's ``pixel_values``
        rows, or ``None`` for text-only episodes.
    """

    token_lengths: torch.Tensor
    pixel_rows: torch.Tensor | None = None


class Trajectory(BaseModel):
    """One completed LLM trajectory.

    :param token_ids: ``(1, T)`` prompt and generated token ids for the whole episode.
    :param action_masks: ``(1, T - 1)`` mask marking action (model-generated) positions.
    :param turn_ids: ``(1, T - 1)`` turn index per action token, ``-1`` elsewhere.
    :param rewards: ``(max_turns,)`` or ``(1, max_turns)`` per-turn rewards.
    :param sampling_logps: Optional 1-D generated-token sampling logprobs.
    :param pixel_values: Optional vision tensor for trainer forward on VL episodes.
    :param segments: Segment layout when the episode context restarted, else ``None``.
    """

    token_ids: torch.Tensor
    action_masks: torch.Tensor
    turn_ids: torch.Tensor
    rewards: torch.Tensor
    sampling_logps: torch.Tensor | None = None
    pixel_values: torch.Tensor | None = None
    segments: EpisodeSegments | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_shapes(self) -> Trajectory:
        """Require the per-action-token fields to be one shorter than the token ids."""
        expected = int(self.token_ids.shape[-1]) - 1
        for name, tensor in (
            ("action_masks", self.action_masks),
            ("turn_ids", self.turn_ids),
        ):
            if int(tensor.shape[-1]) != expected:
                msg = (
                    f"{name} must have length token_ids - 1 ({expected}), "
                    f"got {int(tensor.shape[-1])}."
                )
                raise ValueError(msg)
        if self.segments is not None:
            validate_episode_segments(
                self.segments, int(self.token_ids.shape[-1]), self.pixel_values
            )
        return self


def validate_episode_segments(
    segments: EpisodeSegments,
    token_count: int,
    pixel_values: torch.Tensor | None,
) -> None:
    """Check a segment layout against the episode row it describes.

    :param segments: Segment layout to check.
    :param token_count: Token count of the episode row.
    :param pixel_values: The episode's vision tensor, or ``None``.
    :raises ValueError: If the layout does not tile the row or its vision rows.
    """
    lengths = segments.token_lengths
    if lengths.dim() != 1 or int(lengths.numel()) < 2:
        msg = f"segments.token_lengths must be 1-D with at least 2 segments, got shape {tuple(lengths.shape)}."
        raise ValueError(msg)
    if int(lengths.min()) < 2:
        msg = "Every segment needs at least 2 tokens."
        raise ValueError(msg)
    if int(lengths.sum()) != token_count:
        msg = f"segments.token_lengths sum to {int(lengths.sum())}, episode has {token_count} tokens."
        raise ValueError(msg)
    rows = segments.pixel_rows
    if (rows is None) != (pixel_values is None):
        msg = "segments.pixel_rows must be set exactly when pixel_values is set."
        raise ValueError(msg)
    if rows is None or pixel_values is None:
        return
    if rows.shape != lengths.shape:
        msg = f"segments.pixel_rows shape {tuple(rows.shape)} does not match token_lengths shape {tuple(lengths.shape)}."
        raise ValueError(msg)
    if int(rows.sum()) != int(pixel_values.shape[0]):
        msg = f"segments.pixel_rows sum to {int(rows.sum())}, pixel_values has {int(pixel_values.shape[0])} rows."
        raise ValueError(msg)


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
    """Collated batch built by :func:`collate_rollout_groups`.

    ``token_ids`` and ``action_masks`` are the ragged per-row tensors that
    ``learn()`` pads itself; ``rewards`` and ``turn_ids`` are pre-stacked rectangles.

    :param token_ids: One ``(1, T)`` prompt-plus-generation tensor per trajectory.
    :param action_masks: One ``(1, T - 1)`` tensor per trajectory.
    :param rewards: ``(B, max_turns)`` float tensor of per-turn rewards.
    :param turn_ids: ``(B, T_max - 1)`` tensor padded with ``-1``, or ``None`` when empty.
    :param token_lengths: ``(B,)`` long tensor of per-row sequence lengths.
    :param sampling_logps: Per-row logprob tensors, or ``None`` when none were captured.
    :param pixel_values: Per-row vision tensors, or ``None`` when none were captured.
    :param segments: Per-row segment layouts, or ``None`` when no row restarted.
    """

    token_ids: list[torch.Tensor]
    action_masks: list[torch.Tensor]
    rewards: torch.Tensor
    turn_ids: torch.Tensor | None
    token_lengths: torch.Tensor
    sampling_logps: list[torch.Tensor | None] | None = None
    pixel_values: list[torch.Tensor | None] | None = None
    segments: list[EpisodeSegments | None] | None = None

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def is_empty(self) -> bool:
        """Whether the batch holds no trajectories."""
        return len(self.token_ids) == 0

    def experiences(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Return the ``(token_ids, action_masks, rewards)`` learn tuple."""
        return (self.token_ids, self.action_masks, self.rewards)


def collate_rollout_groups(groups: Sequence[RolloutGroup]) -> LLMExperienceBatch:
    """Flatten groups into one batch, padding turn ids and rewards into rectangles.

    :param groups: Groups to collate, in order.
    :type groups: Sequence[RolloutGroup]
    :return: The collated batch.
    :rtype: LLMExperienceBatch
    """
    trajectories = [traj for group in groups for traj in group.trajectories]
    if not trajectories:
        return LLMExperienceBatch(
            token_ids=[],
            action_masks=[],
            rewards=torch.zeros(0, 0),
            turn_ids=None,
            token_lengths=torch.zeros(0, dtype=torch.long),
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
    pixels = [traj.pixel_values for traj in trajectories]
    segments = [traj.segments for traj in trajectories]
    return LLMExperienceBatch(
        token_ids=[traj.token_ids for traj in trajectories],
        action_masks=[traj.action_masks for traj in trajectories],
        rewards=rewards.float(),
        turn_ids=turn_ids,
        token_lengths=torch.tensor(
            [int(traj.token_ids.shape[-1]) for traj in trajectories],
            dtype=torch.long,
        ),
        sampling_logps=logps if any(lp is not None for lp in logps) else None,
        pixel_values=pixels if any(pv is not None for pv in pixels) else None,
        segments=segments if any(seg is not None for seg in segments) else None,
    )
