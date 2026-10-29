# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""vLLM max_lora_rank is the smallest accepted value covering the adapter rank."""

import pytest

from agilerl.arena.memory.resolve.lora import resolve_max_lora_rank


class TestResolveMaxLoraRank:
    @pytest.mark.parametrize(
        ("adapter_rank", "expected"),
        [(0, 8), (2, 8), (8, 8), (9, 16), (16, 16), (300, 320), (512, 512)],
    )
    def test_rounds_the_adapter_rank_up_to_an_accepted_value(
        self, adapter_rank: int, expected: int
    ) -> None:
        assert resolve_max_lora_rank(adapter_rank) == expected

    def test_rejects_an_adapter_rank_above_every_accepted_value(self) -> None:
        with pytest.raises(
            ValueError,
            match="LoRA adapter rank 513 needs a vLLM max_lora_rank",
        ):
            resolve_max_lora_rank(513)
