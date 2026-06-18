"""CISPO algorithm specification."""

from __future__ import annotations

from agilerl.arena.models.algo import register
from agilerl.arena.models.algorithms.grpo import GRPOSpec


@register()
class CISPOSpec(GRPOSpec):
    """Specification for CISPO algorithm (GRPO with CISPO loss)."""
