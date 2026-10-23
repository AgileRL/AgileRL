# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""A LoRA rank above every accepted vLLM max_lora_rank is rejected."""

import pytest

from agilerl.arena.memory.resolve.lora import resolve_max_lora_rank


class TestResolveMaxLoraRank:
    def test_rejects_an_adapter_rank_above_every_accepted_value(self) -> None:
        # Arrange
        configured = None
        adapter_rank = 513

        # Act / Assert
        with pytest.raises(
            ValueError,
            match="LoRA adapter rank 513 needs a vLLM max_lora_rank",
        ):
            resolve_max_lora_rank(configured, adapter_rank)

    def test_rejects_a_configured_ceiling_above_every_accepted_value(self) -> None:
        # Arrange
        configured = 1024
        adapter_rank = 8

        # Act / Assert
        with pytest.raises(
            ValueError,
            match="configured max_lora_rank 1024 needs a vLLM max_lora_rank",
        ):
            resolve_max_lora_rank(configured, adapter_rank)
