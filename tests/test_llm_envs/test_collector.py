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
