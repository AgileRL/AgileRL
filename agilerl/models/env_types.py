# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Environment type vocabulary.

Its own module rather than part of :mod:`agilerl.typing`: importing that pulls
in torch, tensordict and the protocol hierarchy, and this enum is read where
none of those are wanted (manifest parsing). Keep it dependency-free.
"""

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
