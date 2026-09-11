# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GSPO algorithm specification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agilerl.arena.models.algorithms.grpo import GRPOSpec
from agilerl.arena.models.registry import register


@register()
class GSPOSpec(GRPOSpec):
    """GRPO with the GSPO sequence-level loss."""

    loss_type: Literal["gspo"] = Field(
        default="gspo",
        description="GSPO's sequence-level loss.",
    )
