# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Batched in-process collector over a pool of :class:`RolloutHarness` slots."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from agilerl.llm_envs.harness import RolloutHarness
from agilerl.llm_envs.task_assigner import TaskAssigner, _mix_seed
from agilerl.utils.llm_utils import is_rollout_prompt

__all__ = ["RolloutCollector"]

if TYPE_CHECKING:
    from agilerl.typing import RolloutPrompt


class RolloutCollector:
    """Batched in-process collector over ``batch_size * group_size`` :class:`RolloutHarness` slots.

    Driven lock-step (``reset``/``step``/``get_trajectories``, the colocated path,
    single-caller) or per-episode (``reset_episode``/``step_episode``/
    ``finalize_episode``, the async path — a slot is held from reset to finalize).
    The per-episode methods are synchronous and thread-safe; asyncio callers
    offload them via ``asyncio.to_thread``. A ``step_episode`` whose episode is
    finalized while its env round-trip is in flight raises rather than applying
    the stale result to the slot's next episode.
    """

    def __init__(
        self,
        env_factory: Callable[..., RolloutHarness],
        batch_size: int,
        group_size: int,
        env_config: dict[str, Any] | None = None,
        *,
        io_timeout_s: float | None = 600.0,
        base_seed: int | None = None,
        slot_acquire_timeout_s: float | None = 300.0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """Create ``batch_size * group_size`` independent env wrappers.

        :param env_factory: Factory that builds one :class:`RolloutHarness`.
        :param batch_size: Number of logical batch items.
        :param group_size: Grouped rollouts per batch item.
        :param env_config: Optional kwargs passed to ``env_factory``.
        :param io_timeout_s: Deadline (seconds) for one concurrent round of env round-trips
            (raises ``TimeoutError`` rather than blocking); the only bound on an in-process
            env. ``None`` disables it; keep it above any per-env ``timeout_s``.
        :param base_seed: Base seed for the per-episode path's task assignment (the
            lock-step path takes its seed per ``reset`` call).
        :param slot_acquire_timeout_s: How long ``reset_episode`` waits for a free slot
            before raising ``TimeoutError``; ``None`` waits forever.
        :param rank: This process's data-parallel shard index, handed to the
            :class:`TaskAssigner` (from the runtime, never manifest config).
        :param world_size: Number of data-parallel shards.
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
        self._io_timeout_s = io_timeout_s
        self.envs: list[RolloutHarness] = []
        # Hands each group its task (dataset row / seed); built on first reset.
        self._task_assigner: TaskAssigner | None = None
        # Component key set for ``reward_*`` metrics, frozen when the envs are built.
        self.rubric_component_names: tuple[str, ...] = ()
        self._rank = int(rank)
        self._world_size = int(world_size)
        # --- per-episode state (untouched by the lock-step path) ---
        self._base_seed = int(base_seed) if base_seed is not None else None
        self._slot_acquire_timeout_s = slot_acquire_timeout_s
        self._seed_offset = 0
        # One (seed, row_index) per logical slot, built lazily per rollout window.
        self._assignment: list[tuple[int | None, int | None]] | None = None
        self._free_slots: queue.Queue[int] | None = None
        self._episode_to_slot: dict[str, int] = {}
        # Monotonic per-slot activation tokens: bumped each time a slot is
        # (re)assigned, so a step that outlives its episode is detectable.
        self._slot_activations: list[int] = [0] * self.num_envs
        # Guards the slot queue + episode map (mutated from many caller threads);
        # re-entrant because a failed _ensure_slots calls close() while holding it.
        self._slot_lock = threading.RLock()
        # Serializes tokenizer work and env-state mutation across caller threads;
        # env I/O runs outside it so round-trips still overlap.
        self._tokenizer_lock = threading.Lock()
        self._io_executor: ThreadPoolExecutor | None = None

    @property
    def io_executor(self) -> ThreadPoolExecutor:
        """Shared pool for lock-step env I/O and async per-episode offload."""
        if self._io_executor is None:
            self._io_executor = ThreadPoolExecutor(
                max_workers=max(4, int(self.num_envs) + 1),
                thread_name_prefix="env-io",
            )
        return self._io_executor

    @property
    def _is_initialized(self) -> bool:
        """``True`` once every env slot has been created."""
        return len(self.envs) == self.num_envs

    @property
    def num_epochs(self) -> int:
        """Completed passes over the dataset rows (``0`` when not dataset-backed)."""
        return self._task_assigner.num_epochs if self._task_assigner is not None else 0

    def _window_envs(self) -> list[RolloutHarness]:
        """Harnesses in the current batch/group window, in list order."""
        return self.envs[: self.batch_size * self.group_size]

    def _active_envs(self) -> list[RolloutHarness]:
        """Non-terminal envs in the current window, in their stable list order."""
        return [env for env in self._window_envs() if not env.done]

    def _get_prompts(self) -> list[RolloutPrompt] | None:
        """Observations for active envs, or ``None`` when all are terminal."""
        active = self._active_envs()
        if not active:
            return None
        prompts: list[RolloutPrompt] = []
        for env in active:
            obs = env.current_prompt
            if not is_rollout_prompt(obs):
                msg = "an active env always holds a prompt"
                raise TypeError(msg)
            prompts.append(obs)
        return prompts

    def reset(
        self,
        seed: int | None = None,
    ) -> list[RolloutPrompt] | None:
        """Reset all env wrappers, building them on the first call.

        :meth:`TaskAssigner.assign` picks each group's task; prompts return
        in stable list order. If building envs fails partway, the built ones are closed
        and the batch left empty so a retried ``reset`` starts clean.

        :param seed: Optional base seed for deterministic rollouts.
        :return: Active prompts after reset.
        """
        self._build_envs_and_assigner(seed)
        if self._task_assigner is None:
            msg = "_build_envs_and_assigner builds the assigner"
            raise RuntimeError(msg)
        assignments = self._task_assigner.assign(
            self.batch_size, self.group_size, base_seed=seed
        )
        window = self._window_envs()
        # Phase 1 (concurrent — pure I/O): fetch every env's initial prompt.
        fetches = self._map_env_io(
            [
                partial(env._reset_fetch, env_seed, row_index=row)
                for env, (env_seed, row) in zip(window, assignments, strict=True)
            ],
            envs=window,
        )
        # Phase 2 (sequential — tokenizer): apply each prompt.
        for env, (obs_text, info) in zip(window, fetches, strict=True):
            env._reset_apply(obs_text, info)

        return self._get_prompts()

    def get_rubric_score_means(self) -> dict[str, float]:
        """Mean per-episode component sums over the frozen component key set.

        A component missing from every episode reports ``nan`` rather than ``0``,
        so a never-scored criterion is not logged as a zero reward.
        """
        means: dict[str, float] = {}
        for name in self.rubric_component_names:
            values = [
                env.rubric_score_sums[name]
                for env in self._window_envs()
                if name in env.rubric_score_sums
            ]
            means[name] = sum(values) / len(values) if values else float("nan")
        return means

    def step(
        self,
        token_ids: list[torch.Tensor],
        sampling_logps: list[torch.Tensor | None] | None = None,
    ) -> list[RolloutPrompt] | None:
        """Step each active env with its turn's token sequence.

        :param token_ids: One ``prompt + generation`` tensor per active env.
        :param sampling_logps: vLLM logprobs parallel to ``token_ids``; entries or
            the whole list may be ``None``.
        :return: Next active prompts after stepping.
        """
        active = self._active_envs()
        if len(token_ids) != len(active):
            msg = (
                "Number of token sequences does not match number of active envs: "
                f"{len(token_ids)} != {len(active)}"
            )
            raise RuntimeError(msg)
        if sampling_logps is not None and len(sampling_logps) != len(active):
            msg = (
                "Number of sampling logprobs does not match number of active "
                f"envs: {len(sampling_logps)} != {len(active)}"
            )
            raise RuntimeError(msg)
        slps = sampling_logps if sampling_logps is not None else [None] * len(active)
        # Phase 1 (sequential — tokenizer): decode + record each generation.
        gen_texts = [
            env._step_prepare(full, sampling_logps=slp)
            for env, full, slp in zip(active, token_ids, slps, strict=False)
        ]
        # Phase 2 (concurrent — pure I/O): round-trip each env backend at once.
        results = self._map_env_io(
            [
                partial(env._step_env, gt)
                for env, gt in zip(active, gen_texts, strict=False)
            ],
            envs=active,
        )
        # Phase 3 (sequential — tokenizer): apply each result to its own env.
        for env, result in zip(active, results, strict=False):
            env._step_apply(result)
        return self._get_prompts()

    def _replace_timed_out_slots(self, hung: list[RolloutHarness]) -> None:
        """Close each hung harness and put a fresh factory instance in its slot.

        The abandoned thunk still runs on the discarded client; a later reset
        must not see that object's ``_broken`` flag or closed in-process backend.
        """
        for env in hung:
            index = self.envs.index(env)
            env.close()
            self.envs[index] = self.env_factory(**self.env_config)

    def _map_env_io(
        self,
        thunks: list[Callable[[], Any]],
        *,
        envs: list[RolloutHarness] | None = None,
    ) -> list[Any]:
        """Run each zero-arg thunk on :attr:`io_executor`, returning results in order.

        Bounded by ``io_timeout_s``: on completion the first thunk exception (in
        order) propagates; past the deadline each still-running slot is closed
        and replaced with a fresh harness, then ``TimeoutError`` is raised.
        Hung I/O can only mutate the discarded client. A single thunk runs
        inline only when unbounded.
        """
        if not thunks:
            return []
        if self._io_timeout_s is None and len(thunks) == 1:
            return [thunks[0]()]
        futures = [self.io_executor.submit(thunk) for thunk in thunks]
        completed = wait(futures, timeout=self._io_timeout_s)
        if completed.not_done:
            if envs is not None:
                self._replace_timed_out_slots(
                    [
                        env
                        for future, env in zip(futures, envs, strict=True)
                        if future in completed.not_done
                    ]
                )
            msg = (
                f"{len(completed.not_done)}/{len(thunks)} env round-trips did "
                f"not finish within io_timeout_s={self._io_timeout_s}s; a hung "
                "env or a stalled transport blocked the batch."
            )
            raise TimeoutError(msg)
        return [future.result() for future in futures]

    def close(self) -> None:
        """Close every env wrapper and clear the slot state; a later reset rebuilds."""
        with self._slot_lock:
            self._episode_to_slot.clear()
            self._free_slots = None
        for env in self.envs:
            env.close()
        self.envs = []
        if self._io_executor is not None:
            self._io_executor.shutdown(wait=False, cancel_futures=True)
            self._io_executor = None

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
        """Collect complete episode tensors from all envs, in list order.

        :return: ``(token_ids_list, action_masks_list, all_turn_ids, all_rewards,
            batch_steps, all_sampling_logps)`` where ``batch_steps`` is the summed turn
            count. ``all_sampling_logps`` is ``None`` when no vLLM logprobs were captured,
            else one 1-D tensor per env (``None`` for envs that captured none).
        """
        token_ids_list: list[torch.Tensor] = []
        action_masks_list: list[torch.Tensor] = []
        all_turn_ids: list[torch.Tensor] = []
        all_rewards: list[torch.Tensor] = []
        all_sampling_logps: list[torch.Tensor | None] = []
        batch_steps = 0
        for env in self._window_envs():
            (
                ep_ids,
                action_mask,
                turn_ids,
                turn_rewards_t,
                sampling_logps,
            ) = env.get_episode_data()
            token_ids_list.append(ep_ids)
            action_masks_list.append(action_mask)
            all_turn_ids.append(turn_ids)
            all_rewards.append(turn_rewards_t)
            batch_steps += len(env.turn_boundaries)
            all_sampling_logps.append(sampling_logps)

        return (
            token_ids_list,
            action_masks_list,
            all_turn_ids,
            all_rewards,
            batch_steps,
            # Collapse to a single ``None`` when nothing was captured.
            (
                all_sampling_logps
                if any(logps is not None for logps in all_sampling_logps)
                else None
            ),
        )

    # --- per-episode API (the async path) ------------------------------------

    def set_group_seed(self, group_seed: int) -> None:
        """Set the grouped seed offset. Raises if any episode still holds a slot.

        Group seeds must stay unique across windows; the caller advances
        ``group_seed`` between windows while :attr:`_base_seed` stays fixed
        (see :meth:`TaskAssigner.assign`). Overlapping async groups pass
        ``seed`` on :meth:`reset_episode` instead.

        :param group_seed: Offset mixed into each episode's env seed.
        """
        with self._slot_lock:
            if self._episode_to_slot:
                msg = "cannot set group seed while episodes are still active"
                raise RuntimeError(msg)
            self._seed_offset = int(group_seed)
            self._assignment = None

    def update_rollout_geometry(
        self,
        *,
        rollout_batch_size: int,
        group_size: int,
    ) -> None:
        """Change the window's batch/group split without recreating envs.

        Capacity-capped: the env list built at construction is the slot pool, so the
        new window may not exceed ``num_envs`` episodes.
        """
        new_batch_size = int(rollout_batch_size)
        new_group_size = int(group_size)
        if new_batch_size <= 0:
            msg = f"batch_size must be > 0, got {rollout_batch_size}."
            raise ValueError(msg)
        if new_group_size <= 0:
            msg = f"group_size must be > 0, got {group_size}."
            raise ValueError(msg)
        with self._slot_lock:
            if self._episode_to_slot:
                msg = "cannot update rollout geometry while episodes are still active"
                raise RuntimeError(msg)
        if new_batch_size * new_group_size > self.num_envs:
            msg = (
                f"requested rollout geometry ({new_batch_size} x {new_group_size} = "
                f"{new_batch_size * new_group_size} episodes) exceeds the slot pool "
                f"of {self.num_envs}; growing beyond the initial allocation requires "
                "recreating the collector"
            )
            raise ValueError(msg)
        self.batch_size = new_batch_size
        self.group_size = new_group_size
        self._assignment = None

    def active_episode_count(self) -> int:
        """Count the episodes currently holding a slot (thread-safe)."""
        with self._slot_lock:
            return len(self._episode_to_slot)

    def active_episode_ids(self, max_ids: int = 16) -> list[str]:
        """Snapshot up to ``max_ids`` active episode IDs, for diagnostics (thread-safe)."""
        with self._slot_lock:
            return [
                str(episode_id) for episode_id in list(self._episode_to_slot)[:max_ids]
            ]

    def _build_envs_and_assigner(self, seed: int | None) -> None:
        """Build the env pool once and, with it, the task assigner seeded by ``seed``."""
        if not self._is_initialized:
            try:
                while len(self.envs) < self.num_envs:
                    self.envs.append(self.env_factory(**self.env_config))
            except Exception:
                self.close()
                raise
        if self._task_assigner is None:
            self._task_assigner = TaskAssigner(
                self.envs[0].dataset_size,
                seed=seed,
                rank=self._rank,
                world_size=self._world_size,
            )
            self.rubric_component_names = tuple(self.envs[0].rubric_components)

    def _ensure_slots(self) -> None:
        """Build the envs, the task assigner and the free-slot queue once (thread-safe)."""
        with self._slot_lock:
            if self._free_slots is not None:
                return
            self._build_envs_and_assigner(self._base_seed)
            free_slots: queue.Queue[int] = queue.Queue()
            for slot in range(self.num_envs):
                free_slots.put_nowait(slot)
            self._free_slots = free_slots

    def _episode_assignment(
        self,
        logical_slot: int | None,
    ) -> tuple[int | None, int | None]:
        """The window's ``(seed, row_index)`` for ``logical_slot`` (``(None, None)`` unpinned)."""
        if logical_slot is None:
            return None, None
        with self._slot_lock:
            if self._task_assigner is None:
                msg = "_ensure_slots builds the assigner"
                raise RuntimeError(msg)
            if self._assignment is None:
                self._assignment = self._task_assigner.assign(
                    self.batch_size,
                    self.group_size,
                    base_seed=self._base_seed,
                    seed_offset=self._seed_offset,
                )
            if not 0 <= int(logical_slot) < len(self._assignment):
                msg = (
                    f"logical_slot {logical_slot} is outside the current rollout "
                    f"window of {len(self._assignment)} episodes."
                )
                raise IndexError(msg)
            return self._assignment[int(logical_slot)]

    def _slot_and_activation(self, episode_id: str) -> tuple[int, int]:
        """The ``(slot, activation)`` ``episode_id`` holds; ``KeyError`` when it is not active."""
        with self._slot_lock:
            slot = self._episode_to_slot.get(episode_id)
            if slot is None:
                msg = f"Episode {episode_id!r} is not active."
                raise KeyError(msg)
            return slot, self._slot_activations[slot]

    def _require_current(self, episode_id: str, slot: int, activation: int) -> None:
        """Raise unless ``episode_id`` still holds ``slot`` on the same activation."""
        with self._slot_lock:
            current_slot = self._episode_to_slot.get(episode_id)
            current_activation = self._slot_activations[slot]
        if current_slot != slot or current_activation != activation:
            msg = (
                f"Episode {episode_id!r} no longer owns its env slot: it was "
                "finalized (and the slot possibly reused) while this step was "
                "in flight."
            )
            raise RuntimeError(msg)

    def reset_episode(
        self,
        episode_id: str,
        logical_slot: int | None = None,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Acquire a free slot and reset one episode at its window-assigned task.

        Blocking and thread-safe — the offload target for an asyncio caller. The slot
        is held until :meth:`get_episode_data` / :meth:`finalize_episode`. ``logical_slot``
        indexes the window's group-contiguous ``(seed, row_index)`` assignment; ``None``
        leaves the task unpinned.

        :param episode_id: Caller-unique id naming this episode in later calls.
        :param logical_slot: Position in the rollout window, pinning the group task.
        :param seed: Group-seed offset mixed into this episode's env seed; ``None``
            uses the window assignment from :meth:`set_group_seed`.
        :return: ``(prompt, info)`` — the policy-ready prompt (empty when the
            episode truncated at turn 0).
        """
        self._ensure_slots()
        if self._free_slots is None:
            msg = "an active episode implies slots exist"
            raise RuntimeError(msg)
        assigned_seed, row_index = self._episode_assignment(logical_slot)
        if seed is None:
            env_seed = assigned_seed
        elif self._base_seed is None:
            env_seed = None
        else:
            item = 0 if logical_slot is None else int(logical_slot) // self.group_size
            env_seed = _mix_seed(int(self._base_seed) + int(seed) + item)
        try:
            slot = self._free_slots.get(timeout=self._slot_acquire_timeout_s)
        except queue.Empty:
            msg = (
                f"Timed out after {self._slot_acquire_timeout_s}s acquiring a free env "
                f"slot for episode {episode_id!r}; {self.num_envs} slots are all held "
                "by unfinalized episodes."
            )
            raise TimeoutError(msg) from None
        acquired = False
        try:
            env = self.envs[slot]
            obs_text, info = env._reset_fetch(env_seed, row_index=row_index)
            with self._tokenizer_lock:
                prompt, info = env._reset_apply(obs_text, info)
            with self._slot_lock:
                # Checked under the same lock as the insert, so racing duplicate
                # ids cannot both claim a slot.
                if episode_id in self._episode_to_slot:
                    msg = f"Episode {episode_id!r} is already active."
                    raise RuntimeError(msg)
                self._episode_to_slot[episode_id] = slot
                self._slot_activations[slot] += 1
            acquired = True
            return prompt, info
        finally:
            # A failed reset must release the slot it took so the pool is not starved.
            if not acquired:
                self._free_slots.put_nowait(slot)

    def step_episode(
        self,
        episode_id: str,
        token_ids: torch.Tensor,
        sampling_logps: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance one episode a turn: decode, round-trip its env, apply the result.

        Thread-safe; only the env round-trip runs outside the tokenizer lock, so
        interleaved episodes overlap on I/O and stay serial on tokenizer work.

        :param episode_id: The episode to step (from a prior :meth:`reset_episode`).
        :param token_ids: This turn's ``prompt + generation`` tensor.
        :param sampling_logps: This turn's vLLM sampling logprobs, or ``None``.
        :return: The :meth:`RolloutHarness.step` 5-tuple.
        :raises RuntimeError: If the episode was finalized (and its slot possibly
            reused) while the step was in flight — the stale result is never
            applied to the slot's successor episode.
        """
        slot, activation = self._slot_and_activation(episode_id)
        env = self.envs[slot]
        with self._tokenizer_lock:
            self._require_current(episode_id, slot, activation)
            gen_text = env._step_prepare(token_ids, sampling_logps=sampling_logps)
        env_result = env._step_env(gen_text)
        with self._tokenizer_lock:
            self._require_current(episode_id, slot, activation)
            return env._step_apply(env_result)

    def get_episode_data(
        self,
        episode_id: str,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        """Build one episode's tensors and release its slot.

        :param episode_id: The episode to finalize; ``KeyError`` when not active.
        :return: The :meth:`RolloutHarness.get_episode_data` 5-tuple.
        """
        result = self.finalize_episode(episode_id, missing_ok=False)
        if result is None:
            msg = "missing_ok=False raises instead of returning None"
            raise RuntimeError(msg)
        return result

    def finalize_episode(
        self,
        episode_id: str,
        *,
        missing_ok: bool = True,
    ) -> (
        tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
        ]
        | None
    ):
        """Finalize and release one episode slot exactly once.

        Idempotent when ``missing_ok`` — safe for cancellation cleanup paths that may
        race with normal finalize.
        """
        with self._slot_lock:
            slot = self._episode_to_slot.pop(episode_id, None)
        if slot is None:
            if missing_ok:
                return None
            msg = f"Episode {episode_id!r} is not active."
            raise KeyError(msg)
        try:
            with self._tokenizer_lock:
                return self.envs[slot].get_episode_data()
        finally:
            if self._free_slots is None:
                msg = "an active episode implies slots exist"
                raise RuntimeError(msg)
            self._free_slots.put_nowait(slot)
