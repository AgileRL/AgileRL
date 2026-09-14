# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CISPO algorithm specification."""

from __future__ import annotations

from pydantic import Field

from agilerl.arena.models.algo import register
from agilerl.arena.models.algorithms.grpo import GRPOSpec


@register()
class CISPOSpec(GRPOSpec):
    """Specification for CISPO algorithm (GRPO with CISPO loss)."""

    # CISPO uses asymmetric clip bounds [epsilon_low, epsilon_high].
    clip_coef: float | list[float] = Field(default=0.2)
