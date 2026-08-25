# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class LLMEnvType(str, Enum):
    """Type of LLM environment.

    ``ROLLOUT`` is generative (a ``RolloutHarness``); single-turn reasoning is
    ``max_turns=1``. ``DATASET`` is teacher-forced (a ``DatasetEnv``).
    """

    ROLLOUT = "rollout"
    DATASET = "dataset"

    def __str__(self) -> str:
        return str(self.value)


class EnvSpec(BaseModel):
    """Environment specification for an Arena custom environment.

    :param name: Name of the environment
    :type name: str
    :param num_envs: Number of environments to run in parallel
    :type num_envs: int
    :param version: Version of the environment. Defaults to the latest version.
    :type version: str | None
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    num_envs: int = Field(default=16, ge=1)
    version: str | None = Field(default=None)
