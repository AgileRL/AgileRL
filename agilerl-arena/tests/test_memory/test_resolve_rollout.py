# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Which Ray runs generate on async rollout engines, and how many they start."""

from typing import Any

import pytest

from agilerl.arena.memory.resolve.rollout import (
    apply_run_rollout_mode,
    async_rollout_algorithm,
    rollout_engines_blocker,
    run_rollout_engines,
)
from agilerl.arena.models.algorithms import (
    DPOSpec,
    DQNSpec,
    GRPOSpec,
    LLMPPOSpec,
    LLMREINFORCESpec,
    SFTSpec,
)
from agilerl.arena.models.algorithms.base import AlgorithmSpec


class TestAsyncRolloutAlgorithm:
    @pytest.mark.parametrize("algorithm", [GRPOSpec, LLMPPOSpec, LLMREINFORCESpec])
    def test_rollout_llm_algorithms_run_async(
        self, algorithm: type[AlgorithmSpec]
    ) -> None:
        assert async_rollout_algorithm(algorithm) is True

    @pytest.mark.parametrize("algorithm", [SFTSpec, DPOSpec, DQNSpec])
    def test_dataset_and_classic_algorithms_do_not(
        self, algorithm: type[AlgorithmSpec]
    ) -> None:
        assert async_rollout_algorithm(algorithm) is False


class TestRunRolloutEngines:
    @pytest.mark.parametrize("configured", [2, "auto"])
    def test_an_async_run_keeps_a_set_count(self, configured: int | str) -> None:
        assert run_rollout_engines(True, configured) == configured

    def test_an_async_run_reads_zero_as_auto(self) -> None:
        assert run_rollout_engines(True, 0) == "auto"

    @pytest.mark.parametrize("configured", [0, 2])
    def test_a_run_without_rollout_keeps_the_written_count(
        self, configured: int
    ) -> None:
        assert run_rollout_engines(False, configured) == configured


class TestRolloutEnginesBlocker:
    @pytest.mark.parametrize("configured", [0, 2, "auto"])
    def test_an_async_run_takes_any_engine_count(self, configured: int | str) -> None:
        assert rollout_engines_blocker(True, configured) is None

    def test_a_run_without_rollout_takes_no_engines(self) -> None:
        assert rollout_engines_blocker(False, 0) is None

    @pytest.mark.parametrize("configured", [2, "auto"])
    def test_a_run_without_rollout_rejects_engines(self, configured: int | str) -> None:
        assert rollout_engines_blocker(False, configured) == (
            f"rollout_engines_per_agent={configured} requires async rollout."
        )


class TestApplyRunRolloutMode:
    def test_an_async_run_sets_async_and_auto_engines(self) -> None:
        # Arrange
        training: dict[str, Any] = {"rollout_mode": "colocated", "max_steps": 4}

        # Act
        apply_run_rollout_mode(training, True)

        # Assert
        assert training == {
            "rollout_mode": "async",
            "rollout_engines_per_agent": "auto",
            "max_steps": 4,
        }

    def test_an_async_run_keeps_its_engine_count(self) -> None:
        # Arrange
        training: dict[str, Any] = {"rollout_engines_per_agent": 3}

        # Act
        apply_run_rollout_mode(training, True)

        # Assert
        assert training == {"rollout_mode": "async", "rollout_engines_per_agent": 3}

    def test_a_run_without_rollout_drops_the_rollout_mode(self) -> None:
        # Arrange
        training: dict[str, Any] = {"rollout_mode": "async", "max_steps": 4}

        # Act
        apply_run_rollout_mode(training, False)

        # Assert
        assert training == {"max_steps": 4}
