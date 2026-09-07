# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``agilerl.models.env_types`` re-exports the arena enum."""

from __future__ import annotations

import agilerl.models.env_types as env_types
from agilerl.arena.models.env import LLMEnvType as ArenaLLMEnvType


class TestLLMEnvType:
    def test_reexports_arena_enum(self) -> None:
        assert env_types.LLMEnvType is ArenaLLMEnvType
        assert env_types.__all__ == ["LLMEnvType"]
