# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""LLMEnvType lives on the arena env spec. This module re-exports it.

Hub Ray imports ``agilerl.models.env_types``.
"""

from agilerl.arena.models.env import LLMEnvType

__all__ = ["LLMEnvType"]
