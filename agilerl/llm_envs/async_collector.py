# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Async facade driving ``RolloutCollector``'s per-episode API from an asyncio loop."""

from __future__ import annotations

from asyncio import get_running_loop
from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

from agilerl.llm_envs.env_response import EnvResponse
from agilerl.utils.llm_utils import is_rollout_prompt

T = TypeVar("T")

if TYPE_CHECKING:
    import torch

    from agilerl.llm_envs.collector import RolloutCollector


class AsyncBatchCollector:
    """Drive an in-process :class:`~agilerl.llm_envs.RolloutCollector` from an asyncio loop.

    Per-episode calls are synchronous, so they are offloaded to an executor sized
    to the slot count; the collector's tokenizer lock keeps tokenizer work serial.
    """

    # Finalize does no env I/O; this only guards a saturated executor during cleanup.
    episode_finalize_timeout_s = 30.0

    def __init__(self, collector: RolloutCollector) -> None:
        """Offload per-episode calls onto ``collector``'s I/O executor."""
        self._collector = collector
        self._executor = collector.io_executor

    async def _offload(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run one synchronous collector call on the executor."""
        return await get_running_loop().run_in_executor(
            self._executor,
            partial(fn, *args, **kwargs),
        )

    async def reset(
        self,
        episode_id: str,
        logical_slot_idx: int | None = None,
        *,
        seed: int | None = None,
    ) -> EnvResponse:
        """Acquire a slot and reset one episode; ``done`` when there is no policy prompt."""
        prompt, info = await self._offload(
            self._collector.reset_episode,
            episode_id,
            logical_slot_idx,
            seed=seed,
        )
        # Empty / non-prompt observations (turn-0 overflow) must not be generated from.
        done = not (isinstance(prompt, Mapping) and is_rollout_prompt(prompt))
        return EnvResponse(
            episode_id=episode_id,
            observation=prompt,
            reward=0.0,
            done=done,
            info=info if isinstance(info, dict) else {},
        )

    async def step(
        self,
        episode_id: str,
        token_ids: torch.Tensor,
    ) -> EnvResponse:
        """Advance one episode a turn.

        Sampling logprobs stay with the engine rather than passing through here:
        it accumulates them per turn so a turn that returns none drops the whole
        episode's set, keeping the rest aligned to the action mask.
        """
        prompt, reward, terminated, truncated, info = await self._offload(
            self._collector.step_episode,
            episode_id,
            token_ids,
        )
        return EnvResponse(
            episode_id=episode_id,
            observation=prompt,
            reward=float(reward),
            done=bool(terminated or truncated),
            info=info if isinstance(info, dict) else {},
        )

    async def get_episode_data(
        self,
        episode_id: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Build one episode's tensors and release its slot."""
        return await self._offload(self._collector.get_episode_data, episode_id)

    async def finalize_episode(
        self,
        episode_id: str,
        *,
        missing_ok: bool = True,
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
        ]
        | None
    ):
        """Finalize and release one episode slot exactly once (idempotent)."""
        return await self._offload(
            self._collector.finalize_episode,
            episode_id,
            missing_ok=missing_ok,
        )

    def set_group_seed(self, group_seed: int) -> None:
        """Start a rollout window on the underlying collector."""
        self._collector.set_group_seed(group_seed)

    def update_rollout_geometry(
        self,
        *,
        rollout_batch_size: int,
        group_size: int,
    ) -> None:
        """Change the window's batch/group split without recreating envs."""
        self._collector.update_rollout_geometry(
            rollout_batch_size=rollout_batch_size,
            group_size=group_size,
        )

    def active_episode_count(self) -> int:
        """Count the episodes currently holding a slot."""
        return self._collector.active_episode_count()

    def active_episode_ids(self, max_ids: int = 16) -> list[str]:
        """Snapshot up to ``max_ids`` active episode IDs, for diagnostics."""
        return self._collector.active_episode_ids(max_ids)

    def close(self) -> None:
        """Shut the shared I/O executor and close every slot's env client.

        No wait: a thread hung in a ``/ws`` call would otherwise block engine
        shutdown for the full stop timeout.
        """
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._collector.close()
