# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Environment type vocabulary.

Re-exports :class:`agilerl.arena.models.env.LLMEnvType` so manifest parsing
can import the enum without pulling torch or the protocol hierarchy.
"""

from __future__ import annotations

from agilerl.arena.models.env import LLMEnvType

__all__ = ["LLMEnvType"]
