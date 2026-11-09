# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Stacked-bar memory breakdowns."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

import agilerl.arena.memory.formulas as formulas

PhaseName = Literal["training", "generation"]


class MemoryComponent(BaseModel):
    """One segment of a phase's stacked bar."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    key: str
    label: str
    n_bytes: int = Field(alias="bytes", serialization_alias="bytes")
    detail: dict[str, int] = Field(default_factory=dict)
    note: str | None = None


class PhaseBreakdown(BaseModel):
    """Predicted peak for one phase on one device."""

    model_config = ConfigDict(frozen=True)

    phase: PhaseName
    components: tuple[MemoryComponent, ...]
    device_total_bytes: int
    device_usable_bytes: int
    warnings: tuple[str, ...] = ()
    # Host RAM one training process holds beside the device peak. Not part
    # of ``total_bytes`` or ``fits``.
    host: tuple[MemoryComponent, ...] = ()

    @property
    def total_bytes(self) -> int:
        return sum(c.n_bytes for c in self.components)

    @property
    def host_total_bytes(self) -> int:
        return sum(c.n_bytes for c in self.host)

    @property
    def fits(self) -> bool:
        """The peak is inside the card, and vLLM's start-up pool holds one full sequence."""
        if self.total_bytes > self.device_usable_bytes:
            return False
        kv = next((c for c in self.components if c.key == "kv_cache"), None)
        return (
            kv is None or kv.detail["startup_pool"] >= kv.detail["full_sequence_bytes"]
        )

    @property
    def fits_with_buffer(self) -> bool:
        """Training fit leaves the underprediction share free; generation does not."""
        if self.phase != "training":
            return self.fits
        return formulas.recommendation_fits(self.total_bytes, self.device_usable_bytes)

    @property
    def headroom_bytes(self) -> int:
        return self.device_usable_bytes - self.total_bytes
