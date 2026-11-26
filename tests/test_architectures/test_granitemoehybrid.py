# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Granite 4 hybrid runtime config."""

from __future__ import annotations

from agilerl.architectures.catalog import FAMILY_RUNTIME_CONFIGS


class TestGranitemoehybridRuntimeConfig:
    def test_engine_kwargs_keep_the_checkpoint_layer_types(self) -> None:
        vllm_kwargs = FAMILY_RUNTIME_CONFIGS["granitemoehybrid"].vllm.model_dump(
            exclude_none=True
        )

        assert "hf_overrides" not in vllm_kwargs

    def test_tool_calls_use_the_tool_call_tag_parser(self) -> None:
        assert FAMILY_RUNTIME_CONFIGS["granitemoehybrid"].chat_format == ("tool_call")
