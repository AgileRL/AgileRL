"""Environment type vocabulary, importable without the heavy env-spec deps."""

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
