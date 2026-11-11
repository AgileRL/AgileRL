# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Decode-speed constants are looked up by the rollout GPU's compute capability."""

from __future__ import annotations

import pytest

from agilerl.arena.memory.manifest import lookup_gpu
from agilerl.arena.memory.resolve.decode_costs import (
    MERGED_MOE_PREFILL_SPEEDUP,
    SM80_DECODE_COSTS,
    SM90_DECODE_COSTS,
    SM90_MERGED_SPEEDUPS,
    UNMEASURED_DECODE_COSTS,
    UNMEASURED_MERGED_DENSE_SPEEDUP,
    UNMEASURED_MERGED_SPEEDUPS,
    decode_costs,
    merged_engine_speedups,
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

    @pytest.mark.parametrize(
        "device",
        [
            lookup_gpu("NVIDIA B200").device_spec(),
            DeviceSpec(total_bytes=80 * GiB, compute_capability=(12, 0)),
        ],
        ids=["b200", "sm120"],
    )
    def test_hopper_and_newer_without_a_row_use_the_sm90_entry(
        self, device: DeviceSpec
    ) -> None:
        assert device.compute_capability is not None
        assert device.compute_capability >= (9, 0)
        assert decode_costs(device) == SM90_DECODE_COSTS

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


class TestMergedEngineSpeedups:
    def test_a100_dense_uses_the_measured_token_rate(self) -> None:
        device = lookup_gpu("NVIDIA A100-SXM4-80GB").device_spec()

        speedups = merged_engine_speedups(device)

        assert speedups.dense_decode == pytest.approx(1054 / 406)
        assert speedups.dense_prefill == pytest.approx(1054 / 406)
        assert speedups.moe_prefill == MERGED_MOE_PREFILL_SPEEDUP

    def test_a100_moe_decode_keeps_the_unmeasured_fallback(self) -> None:
        device = lookup_gpu("NVIDIA A100-SXM4-80GB").device_spec()

        speedups = merged_engine_speedups(device)

        assert speedups.moe_decode == UNMEASURED_MERGED_DENSE_SPEEDUP

    def test_h100_uses_the_fitted_token_rates(self) -> None:
        device = lookup_gpu("NVIDIA H100 80GB HBM3").device_spec()

        speedups = merged_engine_speedups(device)

        assert speedups.dense_decode == pytest.approx(2.146)
        assert speedups.dense_prefill == pytest.approx(2.146)
        assert speedups.moe_decode == pytest.approx(1.205)
        assert speedups.moe_prefill == MERGED_MOE_PREFILL_SPEEDUP

    @pytest.mark.parametrize(
        "device",
        [
            lookup_gpu("NVIDIA B200").device_spec(),
            DeviceSpec(total_bytes=80 * GiB, compute_capability=(12, 0)),
        ],
        ids=["b200", "sm120"],
    )
    def test_hopper_and_newer_without_a_row_use_the_sm90_entry(
        self, device: DeviceSpec
    ) -> None:
        assert device.compute_capability is not None
        assert device.compute_capability >= (9, 0)
        assert merged_engine_speedups(device) == SM90_MERGED_SPEEDUPS

    def test_l4_keeps_the_unmeasured_fallback(self) -> None:
        device = lookup_gpu("NVIDIA L4").device_spec()

        assert device.compute_capability == (8, 9)
        assert merged_engine_speedups(device) == UNMEASURED_MERGED_SPEEDUPS

    def test_unknown_capability_keeps_the_unmeasured_fallback(self) -> None:
        device = DeviceSpec(total_bytes=80 * GiB)

        assert merged_engine_speedups(device) == UNMEASURED_MERGED_SPEEDUPS
