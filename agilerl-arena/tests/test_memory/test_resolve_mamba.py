# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Mamba layers take the unfused path only where the fused kernels return NaN."""

import pytest

from agilerl.arena.memory.resolve.mamba import mamba_scan_is_unfused


class TestMambaScanIsUnfused:
    def test_ada_runs_the_unfused_path(self) -> None:
        assert mamba_scan_is_unfused((8, 9)) is True

    @pytest.mark.parametrize("compute_capability", [(8, 0), (8, 6), (9, 0), (10, 0)])
    def test_other_devices_run_the_fused_scan(
        self, compute_capability: tuple[int, int]
    ) -> None:
        assert mamba_scan_is_unfused(compute_capability) is False
