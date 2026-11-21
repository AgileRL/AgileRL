# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Routed-expert LoRA picks the expert GEMM kernel, its rank, and recompute per block."""

import pytest

from agilerl.arena.memory.resolve.moe import (
    expert_gemm_backend,
    grouped_gemm_supported,
    padded_expert_lora_rank,
    resolve_moe_lora_recompute,
)


class TestGroupedGemmSupported:
    @pytest.mark.parametrize("compute_capability", [(9, 0), (10, 0)])
    def test_hopper_and_blackwell_bf16_are_supported(
        self, compute_capability: tuple[int, int]
    ) -> None:
        assert grouped_gemm_supported(compute_capability, "bfloat16") is True

    @pytest.mark.parametrize("compute_capability", [(8, 0), (8, 9), (12, 0)])
    def test_other_architectures_are_unsupported(
        self, compute_capability: tuple[int, int]
    ) -> None:
        assert grouped_gemm_supported(compute_capability, "bfloat16") is False

    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    def test_other_dtypes_are_unsupported(self, dtype: str) -> None:
        assert grouped_gemm_supported((9, 0), dtype) is False


class TestExpertGemmBackend:
    @pytest.mark.parametrize("triton_installed", [True, False])
    def test_blackwell_bf16_takes_grouped_mm(self, triton_installed: bool) -> None:
        assert (
            expert_gemm_backend((10, 0), "bfloat16", triton_installed) == "grouped_mm"
        )

    def test_hopper_without_triton_takes_grouped_mm(self) -> None:
        assert expert_gemm_backend((9, 0), "bfloat16", False) == "grouped_mm"

    @pytest.mark.parametrize(
        ("compute_capability", "dtype"),
        [
            ((8, 0), "bfloat16"),
            ((8, 9), "bfloat16"),
            ((9, 0), "bfloat16"),
            ((9, 0), "float16"),
        ],
    )
    def test_sm80_plus_half_precision_takes_triton(
        self, compute_capability: tuple[int, int], dtype: str
    ) -> None:
        assert expert_gemm_backend(compute_capability, dtype, True) == "triton"

    def test_without_triton_raises_below_hopper(self) -> None:
        with pytest.raises(
            RuntimeError, match=r"no kernel on SM 8\.9 in bfloat16 without Triton"
        ):
            expert_gemm_backend((8, 9), "bfloat16", False)

    @pytest.mark.parametrize(
        ("compute_capability", "dtype"), [((7, 5), "float16"), ((8, 0), "float32")]
    )
    def test_pre_ampere_or_fp32_raises(
        self, compute_capability: tuple[int, int], dtype: str
    ) -> None:
        major, minor = compute_capability
        with pytest.raises(
            RuntimeError, match=rf"no kernel on SM {major}\.{minor} in {dtype}:"
        ):
            expert_gemm_backend(compute_capability, dtype, True)


class TestPaddedExpertLoraRank:
    def test_bf16_rank_rounds_up_to_a_multiple_of_8(self) -> None:
        assert padded_expert_lora_rank(12, 2) == 16

    def test_aligned_rank_is_kept(self) -> None:
        assert padded_expert_lora_rank(16, 2) == 16

    def test_fp32_rank_rounds_up_to_a_multiple_of_4(self) -> None:
        assert padded_expert_lora_rank(6, 4) == 8

    def test_rank_one_pads_to_one_aligned_row(self) -> None:
        assert padded_expert_lora_rank(1, 2) == 8


class TestResolveMoeLoraRecompute:
    def test_recomputes_outside_checkpointed_blocks(self) -> None:
        assert resolve_moe_lora_recompute(checkpointed=False) is True

    def test_keeps_the_graph_inside_checkpointed_blocks(self) -> None:
        assert resolve_moe_lora_recompute(checkpointed=True) is False
