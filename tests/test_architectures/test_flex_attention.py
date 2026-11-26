# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest

from agilerl.architectures.flex_attention import large_head_dim_kernel_options

H100_SHARED_MEMORY = 232448
A100_SHARED_MEMORY = 166912
SPARSE_BLOCK_SIZE = 128
BF16_BYTES = 2


def shared_memory_bytes(options: dict[str, int], head_dim: int) -> dict[str, int]:
    """Pipelined tiles plus held tiles per kernel, as the template loads them."""
    row = 2 * head_dim * BF16_BYTES
    return {
        "forward": options["fwd_num_stages"] * options["BLOCK_N"] * row
        + options["BLOCK_M"] * head_dim * BF16_BYTES,
        "dkdv": (options["bwd_num_stages"] * options["BLOCK_M1"] + options["BLOCK_N1"])
        * row,
        "dq": (options["bwd_num_stages"] * options["BLOCK_N2"] + options["BLOCK_M2"])
        * row,
    }


class TestLargeHeadDimKernelOptions:
    def test_h100_gets_wider_forward_and_deeper_backward_pipeline(self):
        options = large_head_dim_kernel_options(
            512, 512, BF16_BYTES, H100_SHARED_MEMORY
        )

        assert options == {
            "BLOCK_M": 64,
            "BLOCK_N": 32,
            "fwd_num_stages": 2,
            "fwd_num_warps": 8,
            "BLOCK_M1": 16,
            "BLOCK_N1": 32,
            "BLOCK_M2": 32,
            "BLOCK_N2": 16,
            "bwd_num_stages": 3,
            "bwd_num_warps": 8,
        }

    def test_a100_gets_narrower_forward_and_two_stage_backward(self):
        options = large_head_dim_kernel_options(
            512, 512, BF16_BYTES, A100_SHARED_MEMORY
        )

        assert options["BLOCK_N"] == 16
        assert options["bwd_num_stages"] == 2

    @pytest.mark.parametrize("shared_memory", [H100_SHARED_MEMORY, A100_SHARED_MEMORY])
    def test_tiles_fit_shared_memory(self, shared_memory):
        options = large_head_dim_kernel_options(512, 512, BF16_BYTES, shared_memory)

        footprint = shared_memory_bytes(options, 512)

        assert max(footprint.values()) <= shared_memory

    @pytest.mark.parametrize("shared_memory", [H100_SHARED_MEMORY, A100_SHARED_MEMORY])
    def test_tiles_satisfy_the_template_divisibility_rules(self, shared_memory):
        options = large_head_dim_kernel_options(512, 512, BF16_BYTES, shared_memory)

        assert options["BLOCK_N1"] % options["BLOCK_M1"] == 0
        assert options["BLOCK_M2"] % options["BLOCK_N2"] == 0
        for key in (
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_M1",
            "BLOCK_N1",
            "BLOCK_M2",
            "BLOCK_N2",
        ):
            assert SPARSE_BLOCK_SIZE % options[key] == 0

    def test_fp32_accumulators_stay_within_128_registers_per_thread(self):
        options = large_head_dim_kernel_options(
            512, 512, BF16_BYTES, H100_SHARED_MEMORY
        )

        forward = options["BLOCK_M"] * 512 // (32 * options["fwd_num_warps"])
        dkdv = options["BLOCK_N1"] * 2 * 512 // (32 * options["bwd_num_warps"])
        dq = options["BLOCK_M2"] * 512 // (32 * options["bwd_num_warps"])

        assert max(forward, dkdv, dq) <= 128
