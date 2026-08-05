# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Synchronous multi-turn vector environment utilities for LLM rollouts."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from agilerl.typing import ReasoningPrompts
from agilerl.utils.llm_obs import is_reasoning_prompts

if TYPE_CHECKING:
    from agilerl.protocols import TokenizedMultiTurnEnv


def _as_prompt(obs: str | Mapping[str, object]) -> ReasoningPrompts:
    """Read a multi-turn observation as a tokenized prompt.

    :param obs: The observation returned by the wrapped environment.
    :type obs: str | Mapping[str, object]
    :returns: The observation as a tokenized prompt.
    :rtype: ReasoningPrompts
    """
    if isinstance(obs, str):
        msg = "SyncMultiTurnVecEnv expects tokenized prompt observations, got str"
        raise TypeError(msg)
    if not is_reasoning_prompts(obs):
        msg = "SyncMultiTurnVecEnv expects ReasoningPrompts observations"
        raise TypeError(msg)
    return obs


@dataclass
class Trajectory:
    """State for one environment rollout within a synchronized vector batch.

    :param env: The tokenized multi-turn environment this trajectory steps.
    :type env: TokenizedMultiTurnEnv
    :param batch_idx: Index of the logical batch item this trajectory belongs to.
    :type batch_idx: int
    :param group_idx: Index of this trajectory within its group (the buffer holds
        ``batch_size * group_size`` trajectories laid out group-contiguous).
    :type group_idx: int
    :param prompt: The current prompt the environment is rolling out.
    :type prompt: ReasoningPrompts
    :param done: Whether this rollout has terminated.
    :type done: bool
    :param sampling_logps: Per-token sampling logprobs from the vLLM rollout, one
        1-D tensor per turn; ``get_trajectories`` concatenates them across turns.
        Defaults to an empty list.
    :type sampling_logps: list[torch.Tensor]
    """

    env: TokenizedMultiTurnEnv
    batch_idx: int
    group_idx: int
    prompt: ReasoningPrompts
    done: bool
    sampling_logps: list[torch.Tensor] = field(default_factory=list)


class TrajectoryBuffer:
    """Container for synchronized rollout trajectories."""

    def __init__(self, batch_size: int, group_size: int) -> None:
        """Initialize an empty trajectory buffer.

        :param batch_size: Number of logical batch items.
        :type batch_size: int
        :param group_size: Number of grouped trajectories per batch item.
        :type group_size: int
        """
        if batch_size <= 0:
            msg = f"batch_size must be > 0, got {batch_size}."
            raise ValueError(msg)
        if group_size <= 0:
            msg = f"group_size must be > 0, got {group_size}."
            raise ValueError(msg)
        self.batch_size = batch_size
        self.group_size = group_size
        self.trajectories: list[Trajectory] = []

    @property
    def is_initialized(self) -> bool:
        """Return ``True`` when the trajectory buffer is initialized."""
        return len(self.trajectories) == (self.batch_size * self.group_size)

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Append a trajectory to the buffer."""
        self.trajectories.append(trajectory)

    def clear(self) -> None:
        """Remove all stored trajectories."""
        self.trajectories.clear()

    def has_active(self) -> bool:
        """Return ``True`` when at least one trajectory is still active."""
        return any(not trajectory.done for trajectory in self.trajectories)

    def get_prompts(self) -> list[ReasoningPrompts] | None:
        """Return prompt dicts for active trajectories in stable order.

        :return: Active prompt dictionaries sorted by ``(batch_idx, group_idx)``,
            or ``None`` when all trajectories are terminal.
        :rtype: list[ReasoningPrompts] | None
        """
        active_trajectories = self.get_active_trajectories(sorted_by_index=True)
        if len(active_trajectories) == 0:
            return None
        return [trajectory.prompt for trajectory in active_trajectories]

    def get_active_trajectories(
        self,
        *,
        sorted_by_index: bool = False,
    ) -> list[Trajectory]:
        """Get active (non-terminal) trajectories."""
        trajectories = [
            trajectory for trajectory in self.trajectories if not trajectory.done
        ]
        if sorted_by_index:
            trajectories.sort(key=lambda t: (t.batch_idx, t.group_idx))
        return trajectories

    def sort(self, key: Callable[[Trajectory], Any]) -> None:
        """Sort trajectories in place."""
        self.trajectories.sort(key=key)

    def __iter__(self) -> Iterator[Trajectory]:
        """Iterate over stored trajectories."""
        return iter(self.trajectories)

    def reset_trajectory(self, seed: int | None, env_idx: int) -> None:
        """Reset one trajectory in place.

        :param seed: Optional reset seed passed to the wrapped environment.
        :type seed: int | None
        :param env_idx: Index into ``self.trajectories`` to reset.
        :type env_idx: int
        """
        if env_idx < 0 or env_idx >= len(self.trajectories):
            msg = (
                "env_idx out of bounds for trajectory buffer: "
                f"{env_idx} not in [0, {len(self.trajectories) - 1}]"
            )
            raise IndexError(msg)
        prompt_dict, _ = self.trajectories[env_idx].env.reset(seed=seed)
        self.trajectories[env_idx].prompt = _as_prompt(prompt_dict)
        self.trajectories[env_idx].done = False
        self.trajectories[env_idx].sampling_logps.clear()

    def __getitem__(self, index: int) -> Trajectory:
        return self.trajectories[index]

    def __len__(self) -> int:
        return len(self.trajectories)


class SyncMultiTurnVecEnv:
    """Synchronous multi-turn vector environment for LLM rollouts.

    Maintains ``batch_size * group_size`` independent multi-turn environments and
    steps all active trajectories in lock-step using policy completions.
    """

    def __init__(
        self,
        env_factory: Callable[..., TokenizedMultiTurnEnv],
        batch_size: int,
        group_size: int,
        env_config: dict[str, Any] | None = None,
    ) -> None:
        """Create ``batch_size * group_size`` independent environments.

        :param env_factory: Factory that builds one tokenized multi-turn environment.
        :type env_factory: Callable[..., TokenizedMultiTurnEnv]
        :param batch_size: Number of logical batch items.
        :type batch_size: int
        :param group_size: Number of grouped trajectories per batch item.
        :type group_size: int
        :param env_config: Optional kwargs passed to ``env_factory``.
        :type env_config: dict[str, Any] | None
        """
        if batch_size <= 0:
            msg = f"batch_size must be > 0, got {batch_size}."
            raise ValueError(msg)
        if group_size <= 0:
            msg = f"group_size must be > 0, got {group_size}."
            raise ValueError(msg)
        if env_config is None:
            env_config = {}
        self.env_factory = env_factory
        self.env_config = env_config
        self.num_envs = batch_size * group_size
        self.batch_size = batch_size
        self.group_size = group_size
        self.trajectories = TrajectoryBuffer(batch_size, group_size)

    def reset(
        self,
        seed: int | None = None,
    ) -> list[ReasoningPrompts] | None:
        """Reset all environments and initialize trajectories.

        Seeds are assigned per batch row (same seed across groups), then prompts
        are returned in stable ``(batch_idx, group_idx)`` order.

        :param seed: Optional base seed for deterministic rollouts.
        :type seed: int | None
        :return: Active prompt dictionaries after reset.
        :rtype: list[ReasoningPrompts] | None
        """
        seed_base = seed
        for batch_idx in range(self.batch_size):
            batch_seed = None if seed_base is None else seed_base + batch_idx
            for group_idx in range(self.group_size):
                env_idx = batch_idx * self.group_size + group_idx
                if not self.trajectories.is_initialized:
                    env_i = self.env_factory(**self.env_config)
                    prompt_dict, _ = env_i.reset(seed=batch_seed)
                    self.trajectories.add_trajectory(
                        Trajectory(
                            env=env_i,
                            batch_idx=batch_idx,
                            group_idx=group_idx,
                            prompt=_as_prompt(prompt_dict),
                            done=False,
                        )
                    )
                else:
                    self.trajectories.reset_trajectory(env_idx=env_idx, seed=batch_seed)
        return self.trajectories.get_prompts()

    def step(
        self,
        completion_ids: list[torch.Tensor],
        sampling_logps: list[torch.Tensor | None] | None = None,
    ) -> list[ReasoningPrompts] | None:
        """Step each active trajectory with its corresponding completion.

        :param completion_ids: One completion tensor per active trajectory.
        :type completion_ids: list[torch.Tensor]
        :param sampling_logps: Sampling logprobs from vLLM rollout for this
            turn, parallel to ``completion_ids``; entries (or the whole list)
            may be ``None`` when nothing was captured.
        :type sampling_logps: list[torch.Tensor | None] | None
        :return: Next active prompt dictionaries after stepping.
        :rtype: list[ReasoningPrompts] | None
        """
        active = self.trajectories.get_active_trajectories(sorted_by_index=True)
        if len(completion_ids) != len(active):
            msg = (
                "Number of completions does not match number of active trajectories: "
                f"{len(completion_ids)} != {len(active)}"
            )
            raise RuntimeError(msg)
        if sampling_logps is not None:
            if len(sampling_logps) != len(active):
                msg = (
                    "Number of sampling logprobs does not match number of active "
                    f"trajectories: {len(sampling_logps)} != {len(active)}"
                )
                raise RuntimeError(msg)
            for traj, slp in zip(active, sampling_logps, strict=True):
                if slp is not None:
                    traj.sampling_logps.append(slp)
        for traj, completion in zip(active, completion_ids, strict=False):
            full_completion = completion
            if full_completion.dim() == 1:
                full_completion = full_completion.unsqueeze(0)
            next_prompt, _reward, terminated, truncated, _info = traj.env.step(
                full_completion,
            )
            traj.done = bool(terminated or truncated)
            if not traj.done:
                traj.prompt = _as_prompt(next_prompt)
        return self.trajectories.get_prompts()

    def close(self) -> None:
        """Close all underlying environments."""
        seen: set[int] = set()
        for traj in self.trajectories:
            env = traj.env
            env_id = id(env)
            if env_id in seen:
                continue
            seen.add(env_id)
            if hasattr(env, "close"):
                env.close()

    def get_trajectories(
        self,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        int,
        list[torch.Tensor | None] | None,
    ]:
        """Collect complete episode tensors from all trajectories.

        :return: ``(completion_ids_list, action_masks_list, all_turn_ids,
            all_rewards, batch_steps, all_sampling_logps)`` where ``batch_steps``
            is the summed number of recorded turn boundaries across trajectories.
            ``all_sampling_logps`` is ``None`` when no vLLM logprobs were captured
            this rollout; otherwise it holds one 1-D tensor of generated-token
            logprobs per trajectory (concatenated across turns), with ``None`` for
            any trajectory that captured none.
        :rtype: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], int, list[torch.Tensor | None] | None]
        """
        completion_ids_list: list[torch.Tensor] = []
        action_masks_list: list[torch.Tensor] = []
        all_turn_ids: list[torch.Tensor] = []
        all_rewards: list[torch.Tensor] = []
        all_sampling_logps: list[torch.Tensor | None] = []
        batch_steps = 0
        self.trajectories.sort(key=lambda t: (t.batch_idx, t.group_idx))
        for traj in self.trajectories:
            ep_ids, action_mask, turn_ids, turn_rewards_t = traj.env.get_episode_data()
            completion_ids_list.append(ep_ids)
            action_masks_list.append(action_mask)
            all_turn_ids.append(turn_ids)
            all_rewards.append(turn_rewards_t)
            batch_steps += len(getattr(traj.env, "turn_boundaries", []))
            turns = traj.sampling_logps
            all_sampling_logps.append(torch.cat(turns) if turns else None)

        return (
            completion_ids_list,
            action_masks_list,
            all_turn_ids,
            all_rewards,
            batch_steps,
            # Collapse to a single ``None`` when nothing was captured, so the
            # caller needs only an ``is not None`` check (no per-row re-scan).
            (
                all_sampling_logps
                if any(logps is not None for logps in all_sampling_logps)
                else None
            ),
        )
