# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Rollout engines per agent picked from the allocation."""

from agilerl.arena.memory.resolve.engine import resolve_rollout_engines_per_agent


class TestResolveRolloutEnginesPerAgent:
    def test_auto_splits_the_surplus_across_members(self) -> None:
        # Act
        engines = resolve_rollout_engines_per_agent(
            "auto",
            total_gpus=8,
            pop_size=2,
            training_gpus_per_agent=1,
            gpus_per_engine=1,
        )

        # Assert
        assert engines == 3

    def test_auto_rounds_a_partial_engine_down(self) -> None:
        # Act
        engines = resolve_rollout_engines_per_agent(
            "auto",
            total_gpus=8,
            pop_size=2,
            training_gpus_per_agent=1,
            gpus_per_engine=2,
        )

        # Assert
        assert engines == 1

    def test_set_count_is_kept(self) -> None:
        # Act
        engines = resolve_rollout_engines_per_agent(
            2,
            total_gpus=8,
            pop_size=1,
            training_gpus_per_agent=1,
            gpus_per_engine=1,
        )

        # Assert
        assert engines == 2

    def test_auto_without_a_spare_engine_slot_is_zero(self) -> None:
        # Act
        engines = resolve_rollout_engines_per_agent(
            "auto",
            total_gpus=4,
            pop_size=2,
            training_gpus_per_agent=2,
            gpus_per_engine=1,
        )

        # Assert
        assert engines == 0
