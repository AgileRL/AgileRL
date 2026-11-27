# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GPU request refusals for Snowflake pools."""

from __future__ import annotations

from agilerl.snowflake.gpu_fit import gpu_counts_message, pool_gpus
from agilerl.snowflake.profile import GpuPool

NV_M = GpuPool("arena_gpu_GPU_NV_M", 6, 4, 24)


class TestPoolGpus:
    def test_rounds_the_maximum_up_to_whole_nodes(self) -> None:
        assert pool_gpus(NV_M) == 8


class TestGpuCountsMessage:
    def test_a_named_pool_too_small_for_the_count_says_what_it_holds(self) -> None:
        text = gpu_counts_message(12, {"GPU_NV_M": NV_M}, "GPU_NV_M")

        assert text == (
            "The GPU_NV_M pool holds 8 GPUs, so it cannot run 12 GPUs.\n"
            "Valid GPU counts on GPU_NV_M: 4, 8 (multiples of 4, up to the "
            "8 GPUs the pool holds).\n"
            "To run more GPUs, raise the pool's maximum GPUs first.\n"
            "Nothing was started."
        )
