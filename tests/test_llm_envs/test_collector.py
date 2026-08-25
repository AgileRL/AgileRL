# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`~agilerl.llm_envs.collector.RolloutCollector` env I/O isolation."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest
import torch

from agilerl.llm_envs import RolloutCollector
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


class TestEpisodeSeeding:
    def test_an_explicit_seed_without_a_base_seed_leaves_the_env_unseeded(
        self,
    ) -> None:
        """No base seed means no reproducible stream to offset into.

        Mixing the group seed alone would look deterministic while ignoring the
        run's seed entirely, so the env is left to pick its own task.
        """
        collector = _plain_collector(base_seed=None)
        try:
            collector.reset_episode("ep-1", 0, seed=7)

            assert collector.envs[0].seen_seed is None
        finally:
            collector.close()

    def test_an_explicit_seed_with_a_base_seed_mixes_both(self) -> None:
        collector = _plain_collector(base_seed=100)
        try:
            collector.reset_episode("ep-1", 0, seed=7)

            assert collector.envs[0].seen_seed is not None
        finally:
            collector.close()
