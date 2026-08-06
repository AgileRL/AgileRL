# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Environment type vocabulary, importable without the heavy env-spec deps."""

from __future__ import annotations

from enum import Enum


class LLMEnvType(str, Enum):
    """Type of LLM environment.

    ``ROLLOUT`` covers every generative regime (a ``RolloutHarness``): single-turn
    reasoning is simply ``max_turns=1``. ``DATASET`` covers the teacher-forced
    regimes (a ``DatasetEnv``), selected by :attr:`LLMEnvSpec.objective`.
    """

    ROLLOUT = "rollout"
    DATASET = "dataset"

    def __str__(self) -> str:
        return str(self.value)
