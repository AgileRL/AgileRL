# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Decode-speed constants are looked up by the rollout GPU's compute capability."""

from __future__ import annotations

import pytest

from agilerl.arena.memory.manifest import lookup_gpu
from agilerl.arena.memory.resolve.decode_costs import (
    SM80_DECODE_COSTS,
    SM90_DECODE_COSTS,
    UNMEASURED_DECODE_COSTS,
    decode_costs,
)
from agilerl.arena.memory.specs import DeviceSpec, GiB


class TestDecodeCosts:
    @pytest.mark.parametrize(
        ("gpu_type", "expected"),
        [
            ("NVIDIA A100-SXM4-80GB", SM80_DECODE_COSTS),
            ("NVIDIA H100 80GB HBM3", SM90_DECODE_COSTS),
        ],
    )
    def test_measured_gpu_gets_its_entry(self, gpu_type, expected) -> None:
        device = lookup_gpu(gpu_type).device_spec()

        assert decode_costs(device) == expected

    def test_unmeasured_capability_prices_fp8_at_the_bf16_rate(self) -> None:
        device = lookup_gpu("NVIDIA L4").device_spec()

        costs = decode_costs(device)

        assert device.compute_capability == (8, 9)
        assert costs == UNMEASURED_DECODE_COSTS
        assert costs.fp8_weight_read_speedup == 1.0
        assert costs.fp8_moe_weight_read_speedup == 1.0

    def test_unknown_capability_prices_fp8_at_the_bf16_rate(self) -> None:
        device = DeviceSpec(total_bytes=80 * GiB)

        assert decode_costs(device) == UNMEASURED_DECODE_COSTS
