# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Learn-time constants are looked up by the training GPU's compute capability."""

from __future__ import annotations

import pytest

from agilerl.arena.memory.manifest import lookup_gpu
from agilerl.arena.memory.resolve.learn_costs import (
    SM80_LEARN_COSTS,
    SM90_LEARN_COSTS,
    learn_costs,
)
from agilerl.arena.memory.specs import DeviceSpec, GiB


class TestLearnCosts:
    @pytest.mark.parametrize(
        ("gpu_type", "expected"),
        [
            ("NVIDIA A100-SXM4-80GB", SM80_LEARN_COSTS),
            ("NVIDIA H100 80GB HBM3", SM90_LEARN_COSTS),
            ("NVIDIA H200", SM90_LEARN_COSTS),
        ],
    )
    def test_measured_gpu_gets_its_entry(self, gpu_type, expected) -> None:
        device = lookup_gpu(gpu_type).device_spec()

        assert learn_costs(device) == expected

    def test_unmeasured_capability_uses_a100_entry(self) -> None:
        device = lookup_gpu("NVIDIA L4").device_spec()

        assert device.compute_capability == (8, 9)
        assert learn_costs(device) == SM80_LEARN_COSTS

    def test_unknown_capability_uses_a100_entry(self) -> None:
        device = DeviceSpec(total_bytes=80 * GiB)

        assert learn_costs(device) == SM80_LEARN_COSTS
