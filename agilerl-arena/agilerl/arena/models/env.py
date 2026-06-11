from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class LLMEnvType(str, Enum):
    """Type of LLM environment."""

    REASONING = "reasoning"
    PREFERENCE = "preference"
    SFT = "sft"
    MULTITURN = "multiturn"

    def __str__(self) -> str:
        return str(self.value)


class EnvSpec(BaseModel):
    """Environment specification for an Arena custom environment.

    :param name: Name of the environment
    :type name: str
    :param num_envs: Number of environments to run in parallel
    :type num_envs: int
    :param version: Version of the environment
    :type version: str
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    num_envs: int = Field(default=16, ge=1)
    version: str = Field(default="v1")
