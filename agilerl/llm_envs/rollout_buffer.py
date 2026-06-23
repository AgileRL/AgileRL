from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from agilerl.llm_envs.rollout_env import RolloutEnv
    from agilerl.typing import RolloutPrompts


@dataclass
class Trajectory:
    """State for one environment rollout within a synchronized vector batch.

    :param env: The multi-turn environment this trajectory steps.
    :type env: RolloutEnv
    :param batch_idx: Index of the logical batch item this trajectory belongs to.
    :type batch_idx: int
    :param group_idx: Index of this trajectory within its group (the buffer holds
        ``batch_size * group_size`` trajectories laid out group-contiguous).
    :type group_idx: int
    :param prompt: The current prompt the environment is rolling out.
    :type prompt: RolloutPrompts
    :param done: Whether this rollout has terminated.
    :type done: bool
    :param sampling_logps: Per-token sampling logprobs from the vLLM rollout, one
        1-D tensor per turn; ``get_trajectories`` concatenates them across turns.
        Defaults to an empty list.
    :type sampling_logps: list[torch.Tensor]
    """

    env: RolloutEnv
    batch_idx: int
    group_idx: int
    prompt: RolloutPrompts
    done: bool
    sampling_logps: list[torch.Tensor] = field(default_factory=list)


class RolloutBuffer:
    """Container for synchronized rollout trajectories."""

    def __init__(self, batch_size: int, group_size: int):
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

    def get_prompts(self) -> list[RolloutPrompts] | None:
        """Return prompt dicts for active trajectories in stable order.

        :return: Active prompt dictionaries sorted by ``(batch_idx, group_idx)``,
            or ``None`` when all trajectories are terminal.
        :rtype: list[RolloutPrompts] | None
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

    def reset_trajectory(
        self,
        seed: int | None,
        env_idx: int,
        row_index: int | None = None,
    ) -> None:
        """Reset one trajectory in place.

        :param seed: Optional reset seed passed to the wrapped environment.
        :type seed: int | None
        :param env_idx: Index into ``self.trajectories`` to reset.
        :type env_idx: int
        :param row_index: Dataset row chosen by the owning ``BatchRolloutEnv``.
        :type row_index: int | None
        """
        if env_idx < 0 or env_idx >= len(self.trajectories):
            msg = (
                "env_idx out of bounds for trajectory buffer: "
                f"{env_idx} not in [0, {len(self.trajectories) - 1}]"
            )
            raise IndexError(msg)
        env = self.trajectories[env_idx].env
        prompt_dict, _ = (
            env.reset(seed=seed, row_index=row_index)
            if row_index is not None
            else env.reset(seed=seed)
        )
        self.trajectories[env_idx].prompt = prompt_dict
        self.trajectories[env_idx].done = False
        self.trajectories[env_idx].sampling_logps.clear()

    def __getitem__(self, index: int) -> Trajectory:
        return self.trajectories[index]

    def __len__(self) -> int:
        return len(self.trajectories)
