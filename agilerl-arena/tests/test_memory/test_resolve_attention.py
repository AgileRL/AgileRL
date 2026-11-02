# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""The trainer's attention backend follows the model family, then the device."""

import pytest

from agilerl.arena.memory.resolve.attention import (
    FAMILY_ATTN_IMPLEMENTATIONS,
    KERNEL_RULE_MODEL_TYPES,
    resolve_attn_implementation,
)
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO


class TestResolveAttnImplementation:
    @pytest.mark.parametrize(
        ("model_type", "expected"),
        [
            ("nemotron_h", "flash_attention_2"),
            ("nemotron_h_omni", "flash_attention_2"),
            ("gemma4", "flex_attention"),
            ("gemma3_text", "flex_attention"),
            ("gpt_oss", "flex_attention"),
        ],
    )
    @pytest.mark.parametrize("cuda", [True, False])
    def test_family_backend_wins_on_any_device(
        self, model_type: str, expected: str, cuda: bool
    ) -> None:
        assert resolve_attn_implementation(model_type, cuda) == expected

    def test_flash_attention_2_on_cuda(self) -> None:
        assert resolve_attn_implementation("qwen3", True) == "flash_attention_2"

    def test_sdpa_off_cuda(self) -> None:
        assert resolve_attn_implementation("qwen3", False) == "sdpa"

    def test_unknown_model_type_takes_the_kernel_rule(self) -> None:
        assert resolve_attn_implementation(None, True) == "flash_attention_2"


class TestAttnImplementationRegistry:
    def test_every_catalog_model_type_is_registered(self) -> None:
        catalog_types = {
            info.config["model_type"] for info in SUPPORTED_MODEL_INFO.values()
        }

        unregistered = (
            catalog_types - FAMILY_ATTN_IMPLEMENTATIONS.keys() - KERNEL_RULE_MODEL_TYPES
        )

        assert unregistered == set()

    def test_no_model_type_is_registered_twice(self) -> None:
        assert FAMILY_ATTN_IMPLEMENTATIONS.keys() & KERNEL_RULE_MODEL_TYPES == set()
