# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Framework spec names are the arena classes, not copies."""

from __future__ import annotations

from agilerl import models
from agilerl.arena import models as arena_models
from agilerl.models.env import GymEnvSpec
from agilerl.models.hpo import MutationSpec
from agilerl.models.training import TrainingSpec


class TestSpecIdentity:
    def test_algorithm_specs_are_arena_classes(self) -> None:
        for name in ("DQNSpec", "PPOSpec", "GRPOSpec", "CQNSpec"):
            assert getattr(models, name) is getattr(arena_models, name)

    def test_section_specs_are_arena_classes(self) -> None:
        assert GymEnvSpec is arena_models.GymEnvSpec
        assert TrainingSpec is arena_models.TrainingSpec
        assert MutationSpec is arena_models.MutationSpec

    def test_manifest_registry_is_shared(self) -> None:
        assert models.MANIFEST_REGISTRY is arena_models.MANIFEST_REGISTRY
