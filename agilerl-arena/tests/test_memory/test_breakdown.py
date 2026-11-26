# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Stacked-bar breakdown copies."""

from agilerl.arena.memory.breakdown import MemoryComponent, PhaseBreakdown


def breakdown() -> PhaseBreakdown:
    return PhaseBreakdown(
        phase="training",
        components=(
            MemoryComponent(key="weights", label="Weights", bytes=10, detail={"a": 1}),
            MemoryComponent(key="adapters", label="Adapters", bytes=5),
        ),
        device_total_bytes=100,
        device_usable_bytes=90,
        warnings=("warned",),
        host=(MemoryComponent(key="pinned", label="Pinned", bytes=3, detail={"b": 2}),),
        host_usable_bytes=50,
    )


class TestPhaseBreakdownDetached:
    def test_copy_equals_the_original(self):
        # Arrange
        original = breakdown()

        # Act
        copied = original.detached()

        # Assert
        assert copied == original
        assert copied.total_bytes == 15
        assert copied.host_total_bytes == 3

    def test_writes_to_the_copy_details_leave_the_original(self):
        # Arrange
        original = breakdown()
        copied = original.detached()

        # Act
        copied.components[0].detail["a"] = 99
        copied.components[1].detail["new"] = 1
        copied.host[0].detail.clear()

        # Assert
        assert original == breakdown()
