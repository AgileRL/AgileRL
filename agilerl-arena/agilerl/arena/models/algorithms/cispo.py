# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CISPO algorithm specification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agilerl.arena.models.algorithms.grpo import GRPOSpec
from agilerl.arena.models.registry import register


@register()
class CISPOSpec(GRPOSpec):
    """GRPO with the CISPO loss and asymmetric clip bounds."""

    loss_type: Literal["cispo"] = Field(
        default="cispo",
        description="CISPO's clipped importance-sampling loss.",
    )
