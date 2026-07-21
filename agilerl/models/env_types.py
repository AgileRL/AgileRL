"""Environment type vocabulary, importable without the heavy env-spec deps.

:mod:`agilerl.models.env` pulls in gymnasium, pandas, and pettingzoo at import
time, so enums needed by the (eagerly imported) algorithm specs live here.
"""

from __future__ import annotations

from enum import Enum


class LLMEnvType(str, Enum):
    """Type of LLM environment."""

    REASONING = "reasoning"
    PREFERENCE = "preference"
    SFT = "sft"
    MULTITURN = "multiturn"

    def __str__(self) -> str:
        return str(self.value)
