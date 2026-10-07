# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`~agilerl.llm_envs.collector.RolloutCollector` env I/O isolation."""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial

import pytest
import torch

from agilerl.components.llm_rollout_data import EpisodeSegments
from agilerl.llm_envs import RolloutCollector
from agilerl.llm_envs.task_assigner import _mix_seed
from tests.helpers.rollout_doubles import RolloutEnvDoubleMixin


class _FakeRemoteClient:
    """Stand-in for :class:`~agilerl.llm_envs.openenv.RemoteEnvClient`.

    ``close`` clears ``_broken`` the way ``_drop_session`` does. In-flight I/O
    can still set ``_broken`` on this same instance after that.
    """

    def __init__(self) -> None:
        self.closed = False
        self._broken = False
        self.reset_calls = 0

    def reset(self, seed: int | None = None, *, row_index: int | None = None):
        _ = (seed, row_index)
        if self._broken:
            msg = "RemoteEnvClient session is broken after a transport error"
            raise RuntimeError(msg)
        self.reset_calls += 1
        return "prompt", {}

    def close(self) -> None:
        self.closed = True
        self._broken = False


class _SlotEnv(RolloutEnvDoubleMixin):
    """Collector slot backed by a fake remote client; optionally hangs on I/O."""

    dataset_size = 0

    def __init__(self, *, hang_on: str | None = None) -> None:
        self.turn_boundaries: list[int] = []
        self.reset_calls: list[int | None] = []
        self.close_calls = 0
        self.done = False
        self.current_prompt: dict = {}
        self.sampling_logps: list[torch.Tensor] = []
        self.hang_on = hang_on
        self._env_client = _FakeRemoteClient()
        self.io_entered = threading.Event()
        self.release = threading.Event()
        self.io_finished = threading.Event()

    def reset(self, seed: int | None = None):
        self.reset_calls.append(seed)
        self.done = False
        self.current_prompt = {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        return self.current_prompt, {}

    def close(self) -> None:
        self.close_calls += 1
        self._env_client.close()

    def _hang_then_break(self) -> None:
        self.io_entered.set()
        assert self.release.wait(timeout=30.0), "hung I/O was never released"
        self._env_client._broken = True
        self.io_finished.set()

    def _reset_fetch(self, seed: int | None = None, *, row_index: int | None = None):
        if self.hang_on == "reset":
            self._hang_then_break()
            return super()._reset_fetch(seed, row_index=row_index)
        self._env_client.reset(seed=seed, row_index=row_index)
        return super()._reset_fetch(seed, row_index=row_index)

    def _step_env(self, gen_text: str):
        del gen_text
        if self.hang_on == "step":
            self._hang_then_break()
        return


def _factory_then_live(initial: Iterator[_SlotEnv]):
    """Yield ``initial`` slots, then live (non-hanging) replacements."""

    def factory() -> _SlotEnv:
        try:
            return next(initial)
        except StopIteration:
            return _SlotEnv()

    return factory


class TestMapEnvIoTimeout:
    def test_step_timeout_replaces_the_slot_so_reset_uses_a_new_harness(self) -> None:
        # Arrange
        hung = _SlotEnv(hang_on="step")
        collector = RolloutCollector(
            env_factory=_factory_then_live(iter([hung])),
            batch_size=1,
            group_size=1,
            io_timeout_s=0.3,
        )
        try:
            collector.reset()
            token_ids = [hung.current_prompt["input_ids"]]

            # Act
            with pytest.raises(
                TimeoutError, match="did not finish within io_timeout_s"
            ):
                collector.step(token_ids)

            replacement = collector.envs[0]
            hung.release.set()
            assert hung.io_finished.wait(timeout=2.0)

            collector.reset()

            # Assert
            assert hung.io_entered.is_set()
            assert hung.close_calls == 1
            assert hung._env_client.closed is True
            assert hung._env_client._broken is True
            assert replacement is not hung
            assert collector.envs[0] is replacement
            assert replacement._env_client is not hung._env_client
            assert replacement._env_client._broken is False
            assert replacement._env_client.reset_calls >= 1
            assert hung._env_client.reset_calls == 1
            assert collector._io_executor is not None
        finally:
            hung.release.set()
            collector.close()

    def test_reset_timeout_replaces_the_slot_so_the_next_reset_uses_a_new_harness(
        self,
    ) -> None:
        # Arrange
        hung = _SlotEnv(hang_on="reset")
        collector = RolloutCollector(
            env_factory=_factory_then_live(iter([hung])),
            batch_size=1,
            group_size=1,
            io_timeout_s=0.3,
        )
        try:
            # Act
            with pytest.raises(
                TimeoutError, match="did not finish within io_timeout_s"
            ):
                collector.reset()

            replacement = collector.envs[0]
            hung.release.set()
            assert hung.io_finished.wait(timeout=2.0)

            collector.reset()

            # Assert
            assert hung.io_entered.is_set()
            assert hung.close_calls == 1
            assert hung._env_client.closed is True
            assert hung._env_client._broken is True
            assert replacement is not hung
            assert collector.envs[0] is replacement
            assert replacement._env_client._broken is False
            assert replacement._env_client.reset_calls >= 1
            assert hung._env_client.reset_calls == 0
        finally:
            hung.release.set()
            collector.close()

    def test_timeout_replaces_only_the_hung_slot(self) -> None:
        # Arrange
        hung = _SlotEnv(hang_on="step")
        fast = _SlotEnv()
        collector = RolloutCollector(
            env_factory=_factory_then_live(iter([hung, fast])),
            batch_size=2,
            group_size=1,
            io_timeout_s=0.3,
        )
        try:
            collector.reset()
            token_ids = [env.current_prompt["input_ids"] for env in collector.envs]

            # Act
            with pytest.raises(
                TimeoutError, match="did not finish within io_timeout_s"
            ):
                collector.step(token_ids)

            # Assert
            assert hung.io_entered.is_set()
            assert hung.close_calls == 1
            assert fast.close_calls == 0
            assert collector.envs[0] is not hung
            assert collector.envs[1] is fast
        finally:
            hung.release.set()
            collector.close()


class _PlainEnv(RolloutEnvDoubleMixin):
    """Minimal slot env: one prompt, terminates on the first step."""

    dataset_size = 0

    def __init__(self) -> None:
        self.done = False
        self.current_prompt: dict = {}
        self.sampling_logps: list[torch.Tensor] = []
        self.turn_boundaries: list[int] = []
        self.close_calls = 0
        self.episode_data = (
            torch.ones(1, 3, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.long),
            torch.ones(1, dtype=torch.float32),
            None,
            None,
            None,
        )

    def reset(self, seed: int | None = None, *, row_index: int | None = None):
        del row_index
        self.seen_seed = seed
        self.done = False
        self.current_prompt = {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        return self.current_prompt, {}

    def step(self, full_completion, sampling_logps=None):
        del full_completion, sampling_logps
        self.done = True
        self.current_prompt = {}
        return {}, 1.0, True, False, {}

    def get_episode_data(self):
        return self.episode_data

    def close(self) -> None:
        self.close_calls += 1


def _plain_collector(**kwargs) -> RolloutCollector:
    return RolloutCollector(env_factory=_PlainEnv, batch_size=1, group_size=1, **kwargs)


class TestCollectorInvariantGuards:
    """The guards that keep a broken invariant from silently corrupting a rollout.

    Each one narrows a type the rest of the method relies on, so they are
    provoked by putting the collector in the state the guard describes.
    """

    def test_an_active_env_holding_no_prompt_is_rejected(self) -> None:
        collector = _plain_collector()
        try:
            collector.reset()
            # Not done, but holding something that is not a prompt: generating
            # from it would feed the policy an empty batch.
            collector.envs[0].current_prompt = {"unexpected": "shape"}

            with pytest.raises(TypeError, match="always holds a prompt"):
                collector._get_prompts()
        finally:
            collector.close()

    def test_reset_requires_the_assigner_it_just_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        collector = _plain_collector()
        try:
            monkeypatch.setattr(
                collector, "_build_envs_and_assigner", lambda seed: None
            )

            with pytest.raises(RuntimeError, match="builds the assigner"):
                collector.reset()
        finally:
            collector.close()

    def test_episode_assignment_requires_the_assigner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        collector = _plain_collector()
        try:
            monkeypatch.setattr(collector, "_ensure_slots", lambda: None)

            with pytest.raises(RuntimeError, match="builds the assigner"):
                collector._episode_assignment(0)
        finally:
            collector.close()

    def test_reset_episode_requires_the_slot_queue(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        collector = _plain_collector()
        try:
            monkeypatch.setattr(collector, "_ensure_slots", lambda: None)

            with pytest.raises(RuntimeError, match="implies slots exist"):
                collector.reset_episode("ep-1")
        finally:
            collector.close()

    def test_get_episode_data_rejects_a_finalize_that_returned_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        collector = _plain_collector()
        try:
            monkeypatch.setattr(
                collector, "finalize_episode", lambda episode_id, missing_ok: None
            )

            with pytest.raises(RuntimeError, match="missing_ok=False raises"):
                collector.get_episode_data("ep-1")
        finally:
            collector.close()

    def test_finalize_requires_the_slot_queue_before_releasing(self) -> None:
        collector = _plain_collector()
        try:
            collector.reset_episode("ep-1")
            # close() drops the queue; a finalize racing it must say so rather
            # than release a slot into nothing.
            collector._free_slots = None

            with pytest.raises(RuntimeError, match="implies slots exist"):
                collector.finalize_episode("ep-1")
        finally:
            collector._free_slots = None
            collector.close()


def _segmented_episode_data() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, EpisodeSegments
]:
    """A two-segment episode row: segments of 3 and 2 tokens."""
    return (
        torch.ones(1, 5, dtype=torch.long),
        torch.tensor([[True, True, False, True]]),
        torch.tensor([[0, 0, -1, 1]]),
        torch.ones(2, dtype=torch.float32),
        None,
        None,
        EpisodeSegments(token_lengths=torch.tensor([3, 2])),
    )


class TestRolloutCollectorGetTrajectories:
    def test_rejects_an_episode_whose_context_restarted(self) -> None:
        # Arrange
        collector = _plain_collector()
        try:
            collector.reset()
            collector.envs[0].episode_data = _segmented_episode_data()

            # Act / Assert
            with pytest.raises(ValueError, match="per-episode API"):
                collector.get_trajectories()
        finally:
            collector.close()


class TestRolloutCollectorGetEpisodeData:
    def test_returns_the_segment_layout_and_releases_the_slot(self) -> None:
        # Arrange
        collector = _plain_collector()
        try:
            collector.reset_episode("ep-1")
            collector.envs[0].episode_data = _segmented_episode_data()

            # Act
            *_tensors, segments = collector.get_episode_data("ep-1")

            # Assert
            assert segments is not None
            assert segments.token_lengths.tolist() == [3, 2]
            assert collector.active_episode_count() == 0
        finally:
            collector.close()


class TestEpisodeSeeding:
    def test_a_group_seed_without_a_base_seed_leaves_the_env_unseeded(
        self,
    ) -> None:
        """No base seed means no reproducible stream to offset into.

        Mixing the group seed alone would look deterministic while ignoring the
        run's seed entirely, so the env is left to pick its own task.
        """
        collector = _plain_collector(base_seed=None)
        try:
            task = collector.assign_group_task(7)
            collector.reset_episode("ep-1", task=task)

            assert task == (None, None)
            assert collector.envs[0].seen_seed is None
        finally:
            collector.close()

    def test_a_group_seed_with_a_base_seed_mixes_both(self) -> None:
        collector = _plain_collector(base_seed=100)
        try:
            collector.reset_episode("ep-1", task=collector.assign_group_task(7))

            assert collector.envs[0].seen_seed == _mix_seed(107)
        finally:
            collector.close()


class _RowEnv(_PlainEnv):
    """Slot env over an 8-row task list; every slot logs its reset rows to ``resets``."""

    dataset_size = 8

    def __init__(self, resets: list[int | None]) -> None:
        super().__init__()
        self.resets = resets

    def reset(self, seed: int | None = None, *, row_index: int | None = None):
        self.resets.append(row_index)
        return super().reset(seed, row_index=row_index)


def _start_group(collector: RolloutCollector, group_seed: int) -> list[str]:
    """Reset both members of one group at a fresh task; return their episode ids."""
    task = collector.assign_group_task(group_seed)
    episode_ids = [f"g{group_seed}-m{member}" for member in range(2)]
    for episode_id in episode_ids:
        collector.reset_episode(episode_id, task=task)
    return episode_ids


class TestRolloutCollectorAssignGroupTask:
    def test_overlapping_groups_cover_every_row_once_per_cycle(self) -> None:
        # Arrange
        resets: list[int | None] = []
        collector = RolloutCollector(
            env_factory=partial(_RowEnv, resets),
            batch_size=8,
            group_size=2,
            base_seed=3,
        )
        try:
            # Act: each group starts while every earlier group is still live.
            first_cycle = [_start_group(collector, seed) for seed in range(8)]
            live_after_first_cycle = collector.active_episode_count()
            for episode_ids in first_cycle:
                for episode_id in episode_ids:
                    collector.finalize_episode(episode_id)
            for seed in range(8, 16):
                _start_group(collector, seed)

            # Assert
            member_rows = [resets[i : i + 2] for i in range(0, len(resets), 2)]
            group_rows = [rows[0] for rows in member_rows]
            assert live_after_first_cycle == 16
            assert len(member_rows) == 16
            assert all(rows[0] == rows[1] for rows in member_rows)
            assert sorted(group_rows[:8]) == list(range(8))
            assert sorted(group_rows[8:]) == list(range(8))
        finally:
            collector.close()

    def test_adaptive_sampling_favours_rows_whose_groups_were_informative(
        self,
    ) -> None:
        # Arrange
        resets: list[int | None] = []
        collector = RolloutCollector(
            env_factory=partial(_RowEnv, resets),
            batch_size=8,
            group_size=2,
            base_seed=3,
            adaptive_task_sampling=True,
        )
        try:
            for seed in range(8):
                for episode_id in _start_group(collector, seed):
                    collector.finalize_episode(episode_id)
            for row in range(8):
                for _ in range(20):
                    collector.record_group_outcome(row, informative=row == 5)

            # Act
            rows = Counter(
                collector.assign_group_task(seed)[1] for seed in range(8, 808)
            )

            # Assert
            weights = {stats.row: stats.weight for stats in collector.task_row_stats()}
            assert rows[5] > 400
            assert all(0 < rows[row] < 80 for row in range(8) if row != 5)
            assert weights[5] > 0.9
            assert all(weights[row] < 0.09 for row in range(8) if row != 5)
        finally:
            collector.close()

    def test_task_row_stats_is_empty_before_the_first_reset(self) -> None:
        collector = RolloutCollector(
            env_factory=partial(_RowEnv, []),
            batch_size=1,
            group_size=1,
            adaptive_task_sampling=True,
        )

        assert collector.task_row_stats() == []

    def test_returns_the_mixed_seed_and_no_row_for_a_procedural_env(self) -> None:
        collector = _plain_collector(base_seed=100)
        try:
            task = collector.assign_group_task(7)

            assert task == (_mix_seed(107), None)
        finally:
            collector.close()

    def test_reset_episode_rejects_a_logical_slot_with_a_task(self) -> None:
        collector = _plain_collector(base_seed=100)
        try:
            task = collector.assign_group_task(7)

            with pytest.raises(ValueError, match="logical_slot or task, not both"):
                collector.reset_episode("ep-1", 0, task=task)
            assert collector.active_episode_count() == 0
        finally:
            collector.close()


def _adaptive_row_collector() -> RolloutCollector:
    return RolloutCollector(
        env_factory=partial(_RowEnv, []),
        batch_size=1,
        group_size=1,
        base_seed=3,
        adaptive_task_sampling=True,
    )


class TestRolloutCollectorTaskSamplerState:
    def test_reports_the_outcomes_fed_back_by_finished_groups(self) -> None:
        # Arrange
        collector = _adaptive_row_collector()
        try:
            collector.assign_group_task(0)

            # Act
            collector.record_group_outcome(5, informative=True)

            # Assert
            assert collector.task_sampler_state() == [
                {"row": 5, "informative": 1.0, "observed": 1.0}
            ]
        finally:
            collector.close()

    def test_is_empty_before_the_first_reset(self) -> None:
        assert _adaptive_row_collector().task_sampler_state() == []


class TestRolloutCollectorLoadTaskSamplerState:
    def test_restores_counts_before_the_first_reset(self) -> None:
        # Arrange
        state = [
            {"row": 2, "informative": 0.0, "observed": 6.0},
            {"row": 5, "informative": 6.0, "observed": 6.0},
        ]
        collector = _adaptive_row_collector()
        try:
            # Act
            collector.load_task_sampler_state(state)

            # Assert
            weights = {stats.row: stats.weight for stats in collector.task_row_stats()}
            rows = Counter(collector.assign_group_task(seed)[1] for seed in range(400))
            assert collector.task_sampler_state() == state
            assert weights[2] == 1 / 8
            assert weights[5] == 7 / 8
            assert rows[5] > 3 * rows[2]
        finally:
            collector.close()


class _SplitEnv(_PlainEnv):
    """Slot env with a 10-row training split and a 4-row held-out split."""

    def __init__(self) -> None:
        super().__init__()
        self.in_eval = False
        self.resets: list[tuple[int | None, int | None, bool]] = []

    @property
    def dataset_size(self) -> int:
        return 4 if self.in_eval else 10

    @contextmanager
    def eval_mode(self) -> Iterator[None]:
        self.in_eval = True
        try:
            yield
        finally:
            self.in_eval = False

    def reset(self, seed: int | None = None, *, row_index: int | None = None):
        self.resets.append((seed, row_index, self.in_eval))
        return super().reset(seed, row_index=row_index)


def _split_collector() -> RolloutCollector:
    return RolloutCollector(env_factory=_SplitEnv, batch_size=1, group_size=1)


class TestRolloutCollectorResetEvalEpisode:
    def test_resets_the_held_out_row_in_eval_mode(self) -> None:
        collector = _split_collector()
        try:
            prompt, _info = collector.reset_eval_episode("eval-1", 2)

            assert collector.envs[0].resets == [(None, 2, True)]
            assert torch.equal(prompt["input_ids"], torch.ones(1, 3, dtype=torch.long))
            assert collector.active_episode_ids() == ["eval-1"]
        finally:
            collector.close()

    def test_a_later_training_reset_uses_the_training_split(self) -> None:
        collector = _split_collector()
        try:
            collector.reset_eval_episode("eval-1", 0)
            collector.finalize_episode("eval-1")

            collector.reset_episode("ep-1", 0)

            assert [in_eval for _, _, in_eval in collector.envs[0].resets] == [
                True,
                False,
            ]
            assert collector.envs[0].dataset_size == 10
        finally:
            collector.close()


class TestRolloutCollectorEvalTaskCount:
    def test_reports_the_held_out_row_count(self) -> None:
        collector = _split_collector()
        try:
            count = collector.eval_task_count()

            assert count == 4
            assert collector.envs[0].dataset_size == 10
        finally:
            collector.close()

    def test_rejects_while_an_episode_is_active(self) -> None:
        collector = _split_collector()
        try:
            collector.reset_episode("ep-1", 0)

            with pytest.raises(RuntimeError, match="episodes are still active"):
                collector.eval_task_count()
        finally:
            collector.close()


class TestRolloutCollectorDpBatch:
    """Global ``batch_size`` is split across data-parallel ranks."""

    def test_splits_prompt_groups_evenly_across_ranks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agilerl.llm_envs.collector.get_world_size", lambda: 2)
        monkeypatch.setattr("agilerl.llm_envs.collector.get_rank", lambda: 1)
        collector = RolloutCollector(
            env_factory=_PlainEnv,
            batch_size=8,
            group_size=4,
        )
        try:
            collector.reset(seed=0)

            assert collector.batch_size == 4
            assert collector.num_envs == 16
            assert len(collector.envs) == 16
        finally:
            collector.close()

    def test_single_rank_keeps_the_full_batch(self) -> None:
        collector = RolloutCollector(
            env_factory=_PlainEnv,
            batch_size=8,
            group_size=4,
        )
        try:
            assert collector.batch_size == 8
            assert collector.num_envs == 32
        finally:
            collector.close()

    def test_rejects_batch_not_divisible_by_world_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agilerl.llm_envs.collector.get_world_size", lambda: 3)
        monkeypatch.setattr("agilerl.llm_envs.collector.get_rank", lambda: 0)
        with pytest.raises(ValueError, match="divisible by the data-parallel"):
            RolloutCollector(
                env_factory=_PlainEnv,
                batch_size=8,
                group_size=1,
            )

    def test_explicit_rank_overrides_get_rank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agilerl.llm_envs.collector.get_world_size", lambda: 2)
        monkeypatch.setattr("agilerl.llm_envs.collector.get_rank", lambda: 0)
        collector = RolloutCollector(
            env_factory=_PlainEnv,
            batch_size=8,
            group_size=4,
            rank=1,
        )
        try:
            collector.reset(seed=0)

            assert collector._task_assigner is not None
            assert collector._task_assigner.rank == 1
        finally:
            collector.close()

    def test_explicit_world_size_overrides_get_world_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agilerl.llm_envs.collector.get_world_size", lambda: 4)
        monkeypatch.setattr("agilerl.llm_envs.collector.get_rank", lambda: 0)
        collector = RolloutCollector(
            env_factory=_PlainEnv,
            batch_size=8,
            group_size=4,
            rank=0,
            world_size=2,
        )
        try:
            collector.reset(seed=0)

            assert collector.batch_size == 4
            assert collector.num_envs == 16
            assert collector._task_assigner is not None
            assert collector._task_assigner.world_size == 2
        finally:
            collector.close()

    def test_rejects_rank_outside_world_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agilerl.llm_envs.collector.get_world_size", lambda: 2)
        with pytest.raises(ValueError, match="rank must be in"):
            RolloutCollector(
                env_factory=_PlainEnv,
                batch_size=8,
                group_size=1,
                rank=2,
            )

    def test_rejects_world_size_below_one(self) -> None:
        with pytest.raises(ValueError, match="world_size must be >= 1"):
            RolloutCollector(
                env_factory=_PlainEnv,
                batch_size=8,
                group_size=1,
                world_size=0,
            )
