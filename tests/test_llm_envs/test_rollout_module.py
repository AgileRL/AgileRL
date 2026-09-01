# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the :mod:`agilerl.llm_envs.rollout` re-export surface.

Docstrings and the docs cross-reference these names by their
``agilerl.llm_envs.rollout`` path (``:class:`~agilerl.llm_envs.rollout.RolloutHarness```),
so a name that stops being re-exported here breaks those references silently.
"""

from __future__ import annotations

import agilerl.llm_envs.rollout as rollout_module


def test_every_exported_name_is_importable() -> None:
    for name in rollout_module.__all__:
        assert hasattr(rollout_module, name), name


def test_exports_are_the_canonical_objects() -> None:
    from agilerl.llm_envs.collector import RolloutCollector
    from agilerl.llm_envs.harness import RolloutHarness
    from agilerl.llm_envs.observation import (
        DEFAULT_OBSERVATION_ROLE,
        OBSERVATION_ROLES,
        observation_role,
        process_observation,
    )
    from agilerl.llm_envs.task_assigner import TaskAssigner

    assert rollout_module.RolloutCollector is RolloutCollector
    assert rollout_module.RolloutHarness is RolloutHarness
    assert rollout_module.TaskAssigner is TaskAssigner
    assert rollout_module.observation_role is observation_role
    assert rollout_module.process_observation is process_observation
    assert rollout_module.DEFAULT_OBSERVATION_ROLE == DEFAULT_OBSERVATION_ROLE
    assert rollout_module.OBSERVATION_ROLES == OBSERVATION_ROLES


def test_mix_seed_is_re_exported_for_the_seeding_helpers() -> None:
    from agilerl.llm_envs.task_assigner import _mix_seed

    assert rollout_module._mix_seed is _mix_seed
