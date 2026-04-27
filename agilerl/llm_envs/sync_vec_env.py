"""Synchronous multi-turn vector environment utilities for LLM rollouts."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import torch

from agilerl.protocols import MultiTurnEnv
from agilerl.typing import ReasoningPrompts
from agilerl.utils.algo_utils import stack_and_pad_experiences


@dataclass
class Trajectory:
    """State for one environment rollout within a synchronized vector batch."""

    env: MultiTurnEnv
    batch_idx: int
    group_idx: int
    prompt: ReasoningPrompts
    done: bool


class TrajectoryBuffer:
    """Container for synchronized rollout trajectories."""

    def __init__(self, batch_size: int, group_size: int):
        """Initialize an empty trajectory buffer."""
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

    def get_prompts(self) -> ReasoningPrompts | None:
        """Return stacked prompts for active trajectories."""
        active_trajectories = self.get_active_trajectories(sorted_by_index=True)
        if len(active_trajectories) == 0:
            return None
        return TrajectoryBuffer._stack_active_prompts(active_trajectories)

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
        """Reset one trajectory in place."""
        if env_idx < 0 or env_idx >= len(self.trajectories):
            msg = (
                "env_idx out of bounds for trajectory buffer: "
                f"{env_idx} not in [0, {len(self.trajectories) - 1}]"
            )
            raise IndexError(msg)
        prompt_dict, _ = self.trajectories[env_idx].env.reset(seed=seed)
        self.trajectories[env_idx].prompt = prompt_dict
        self.trajectories[env_idx].done = False

    @staticmethod
    def _stack_active_prompts(trajectories: list[Trajectory]) -> ReasoningPrompts:
        """Stack prompt fields across active trajectories."""
        if len(trajectories) == 0:
            msg = "Cannot stack prompts from an empty trajectory list."
            raise ValueError(msg)
        prompt_batch = [trajectory.prompt for trajectory in trajectories]
        stacked = cast("ReasoningPrompts", {})
        tensor_keys = (
            "input_ids",
            "attention_mask",
            "trajectory_input_ids",
            "trajectory_attention_mask",
            "stitch_prefix_ids",
        )
        for key in tensor_keys:
            values = [prompt.get(key) for prompt in prompt_batch]
            if all(v is None for v in values):
                continue
            if any(v is None for v in values):
                msg = (
                    f"Inconsistent prompt field '{key}' across active trajectories; "
                    "field must be present for all trajectories or none."
                )
                raise ValueError(msg)
            (stacked_tensor,) = stack_and_pad_experiences(values, padding_values=[0])
            if "attention_mask" in key:
                stacked_tensor = stacked_tensor.long()
            stacked[key] = stacked_tensor

        for required_key in ("input_ids", "attention_mask"):
            if required_key not in stacked:
                msg = f"Missing required prompt field '{required_key}'."
                raise ValueError(msg)

        initial_prompt_lengths = [
            prompt.get("initial_prompt_len") for prompt in prompt_batch
        ]
        if all(v is not None for v in initial_prompt_lengths):
            stacked["initial_prompt_len"] = torch.tensor(
                initial_prompt_lengths,
                dtype=torch.long,
            )

        return stacked

    def __getitem__(self, index: int) -> Trajectory:
        return self.trajectories[index]

    def __len__(self) -> int:
        return len(self.trajectories)


class SyncMultiTurnVecEnv:
    """Synchronous multi-turn vector environment for LLM rollouts."""

    def __init__(
        self,
        env_factory: Callable[..., MultiTurnEnv],
        batch_size: int,
        group_size: int,
        env_config: dict[str, Any] | None = None,
    ):
        """Create ``batch_size * group_size`` independent environments."""
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
    ) -> ReasoningPrompts | None:
        """Reset all environments and initialize trajectories."""
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
                            prompt=prompt_dict,
                            done=False,
                        )
                    )
                else:
                    self.trajectories.reset_trajectory(env_idx=env_idx, seed=batch_seed)
        return self.trajectories.get_prompts()

    def step(self, completion_ids: list[torch.Tensor]) -> ReasoningPrompts | None:
        """Step each active trajectory with its corresponding completion."""
        active = self.trajectories.get_active_trajectories(sorted_by_index=True)
        if len(completion_ids) != len(active):
            msg = (
                "Number of completions does not match number of active trajectories: "
                f"{len(completion_ids)} != {len(active)}"
            )
            raise RuntimeError(msg)
        for traj, completion in zip(active, completion_ids, strict=False):
            full_completion = completion
            if full_completion.dim() == 1:
                full_completion = full_completion.unsqueeze(0)
            next_prompt, _reward, terminated, truncated, _info = traj.env.step(
                full_completion,
            )
            traj.done = bool(terminated or truncated)
            if not traj.done:
                traj.prompt = next_prompt
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
    ]:
        """Collect complete episode tensors from all trajectories."""
        completion_ids_list: list[torch.Tensor] = []
        action_masks_list: list[torch.Tensor] = []
        all_turn_ids: list[torch.Tensor] = []
        all_rewards: list[torch.Tensor] = []
        batch_steps = 0
        self.trajectories.sort(key=lambda t: (t.batch_idx, t.group_idx))
        for traj in self.trajectories:
            ep_ids, action_mask, turn_ids, turn_rewards_t = traj.env.get_episode_data()
            completion_ids_list.append(ep_ids)
            action_masks_list.append(action_mask)
            all_turn_ids.append(turn_ids)
            all_rewards.append(turn_rewards_t)
            batch_steps += len(getattr(traj.env, "turn_boundaries", []))

        return (
            completion_ids_list,
            action_masks_list,
            all_turn_ids,
            all_rewards,
            batch_steps,
        )
