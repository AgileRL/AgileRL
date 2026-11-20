# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the memory estimator tests."""

import math
from collections.abc import Callable

import pytest

from agilerl.arena.memory.resolve import split


@pytest.fixture(autouse=True)
def uncoloured_report(monkeypatch):
    """Rich forces colour on captured output when FORCE_COLOR is set; tests read plain text."""
    monkeypatch.delenv("FORCE_COLOR", raising=False)


@pytest.fixture
def pin_turn_tokens(monkeypatch) -> Callable[[float], None]:
    """Make every single-turn run without expected lengths generate a given tokens per turn, reasoning or not."""

    def pin(tokens: float) -> None:
        monkeypatch.setattr(split, "DEFAULT_REASONING_TURN_TOKENS", tokens)
        monkeypatch.setattr(split, "DEFAULT_TURN_TOKENS", tokens)

    return pin


@pytest.fixture
def cap_filling_turns(pin_turn_tokens: Callable[[float], None]) -> None:
    """Run every single-turn completion to its cap."""
    pin_turn_tokens(math.inf)
