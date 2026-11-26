# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Sequence packing applies only on a backend that isolates packed documents."""

import pytest

from agilerl.arena.memory.resolve.packing import (
    leaks_recurrent_state,
    sequence_packing_blocker,
)


class TestSequencePackingBlocker:
    @pytest.mark.parametrize(
        "attn_implementation", ["flash_attention_2", "flex_attention"]
    )
    def test_one_row_packs_on_a_sparse_backend(self, attn_implementation: str) -> None:
        assert (
            sequence_packing_blocker(
                True, attn_implementation, 1, False, False, "qwen3"
            )
            is None
        )

    @pytest.mark.parametrize(
        "attn_implementation", ["flash_attention_2", "flex_attention"]
    )
    def test_text_model_packs_stacked_rows(self, attn_implementation: str) -> None:
        assert (
            sequence_packing_blocker(
                True, attn_implementation, 2, False, False, "qwen3"
            )
            is None
        )

    @pytest.mark.parametrize(
        ("attn_implementation", "row_copies"),
        [("flash_attention_2", 1), ("flex_attention", 2)],
    )
    def test_vision_language_model_packs_where_layers_isolate_rows(
        self, attn_implementation: str, row_copies: int
    ) -> None:
        assert (
            sequence_packing_blocker(
                True, attn_implementation, row_copies, False, True, "gemma4"
            )
            is None
        )

    @pytest.mark.parametrize("model_type", ["gpt_oss", "granitemoehybrid"])
    def test_position_blind_mask_packs_on_flash_attention_2(
        self, model_type: str
    ) -> None:
        assert (
            sequence_packing_blocker(
                True, "flash_attention_2", 1, False, False, model_type
            )
            is None
        )

    def test_padded_algorithm_does_not_pack(self) -> None:
        # Act
        reason = sequence_packing_blocker(
            False, "flash_attention_2", 1, False, False, "qwen3"
        )

        # Assert
        assert reason == "this algorithm's gradient forward is padded"

    def test_flash_attention_2_does_not_pack_stacked_vision_language_rows(
        self,
    ) -> None:
        # Act
        reason = sequence_packing_blocker(
            True, "flash_attention_2", 2, False, True, "qwen3_5"
        )

        # Assert
        assert reason == (
            "flash_attention_2 derives a vision-language forward's packed "
            "offsets from one row; this forward stacks 2"
        )

    @pytest.mark.parametrize("attn_implementation", ["sdpa", "eager", None])
    def test_dense_backend_does_not_pack(self, attn_implementation: str | None) -> None:
        # Act
        reason = sequence_packing_blocker(
            True, attn_implementation, 1, False, False, "qwen3"
        )

        # Assert
        assert reason == (
            f"attn_implementation {attn_implementation!r} has no varlen or "
            "block-sparse path"
        )

    @pytest.mark.parametrize("model_type", ["gpt_oss", "granitemoehybrid"])
    def test_position_blind_mask_does_not_pack_on_flex_attention(
        self, model_type: str
    ) -> None:
        # Act
        reason = sequence_packing_blocker(
            True, "flex_attention", 1, False, False, model_type
        )

        # Assert
        assert reason == (
            f"{model_type} builds its flex attention masks without position_ids, "
            "so packed documents would attend to each other"
        )

    def test_leaking_recurrent_state_does_not_pack(self) -> None:
        # Act
        reason = sequence_packing_blocker(
            True, "flash_attention_2", 1, True, False, "jamba"
        )

        # Assert
        assert reason == "recurrent mixers do not reset at packed document boundaries"


class TestLeaksRecurrentState:
    @pytest.mark.parametrize(
        "model_type", ["granitemoehybrid", "nemotron_h", "qwen3_5", "qwen3_5_moe"]
    )
    def test_boundary_resetting_family_does_not_leak(self, model_type: str) -> None:
        assert leaks_recurrent_state(model_type, True) is False

    def test_other_hybrid_family_leaks(self) -> None:
        assert leaks_recurrent_state("jamba", True) is True

    def test_model_without_recurrent_layers_does_not_leak(self) -> None:
        assert leaks_recurrent_state("granitemoehybrid", False) is False
