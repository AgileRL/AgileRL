# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GSPO algorithm specification."""

from __future__ import annotations

from agilerl.arena.models.algorithms.gspo import GSPOSpec as ArenaGSPOSpec
from agilerl.models.algo import register
from agilerl.models.algorithms.grpo import GRPOSpec


@register()
class GSPOSpec(GRPOSpec, ArenaGSPOSpec):
    """Specification for GSPO algorithm (GRPO with GSPO loss)."""
