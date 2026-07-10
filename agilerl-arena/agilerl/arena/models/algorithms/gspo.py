"""GSPO algorithm specification."""

from __future__ import annotations

from agilerl.arena.models.algo import register
from agilerl.arena.models.algorithms.grpo import GRPOSpec


@register()
class GSPOSpec(GRPOSpec):
    """Specification for GSPO algorithm (GRPO with GSPO loss)."""
