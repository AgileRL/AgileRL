# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for multimodal vLLM engine kwargs on VL checkpoints."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agilerl.architectures.catalog import FAMILY_RUNTIME_CONFIGS
from agilerl.architectures.runtime import ModelRuntimeConfig
from agilerl.architectures.vllm_language import (
    apply_multimodal_engine_kwargs,
    nested_language_config,
)


class TestNestedLanguageConfig:
    def test_text_config_wins(self) -> None:
        language = SimpleNamespace(architectures=["Qwen2ForCausalLM"])
        config = SimpleNamespace(text_config=language, llm_config=SimpleNamespace())

        assert nested_language_config(config) is language

    def test_llm_config_when_text_config_is_absent(self) -> None:
        language = SimpleNamespace(architectures=["NemotronHForCausalLM"])
        config = SimpleNamespace(llm_config=language)

        assert nested_language_config(config) is language

    def test_none_text_config_uses_llm_config(self) -> None:
        language = SimpleNamespace(architectures=["NemotronHForCausalLM"])
        config = SimpleNamespace(text_config=None, llm_config=language)

        assert nested_language_config(config) is language

    def test_plain_language_config_is_unchanged(self) -> None:
        config = SimpleNamespace(architectures=["LlamaForCausalLM"])

        assert nested_language_config(config) is config


class TestApplyMultimodalEngineKwargs:
    @pytest.mark.parametrize("family", ["nemotron_h_omni", "gemma4"])
    def test_strip_all_serves_language_model_only(self, family: str) -> None:
        # Arrange
        kwargs: dict[str, object] = {}

        # Act
        apply_multimodal_engine_kwargs(
            kwargs,
            strip_multimodal_towers=True,
            runtime=FAMILY_RUNTIME_CONFIGS[family],
        )

        # Assert
        assert kwargs == {"language_model_only": True}

    def test_modality_list_zeroes_those_limits_and_keeps_tower_lora(self) -> None:
        # Arrange
        kwargs: dict[str, object] = {"limit_mm_per_prompt": {"image": 4}}

        # Act
        apply_multimodal_engine_kwargs(
            kwargs,
            strip_multimodal_towers=["audio"],
            runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"],
        )

        # Assert
        assert kwargs == {
            "limit_mm_per_prompt": {"image": 4, "audio": 0},
            "enable_tower_connector_lora": True,
        }

    def test_kept_towers_enable_tower_lora_when_family_supports_it(self) -> None:
        kwargs: dict[str, object] = {}

        apply_multimodal_engine_kwargs(
            kwargs,
            strip_multimodal_towers=False,
            runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"],
        )

        assert kwargs == {"enable_tower_connector_lora": True}

    @pytest.mark.parametrize(
        "runtime",
        [FAMILY_RUNTIME_CONFIGS["gemma4"], ModelRuntimeConfig()],
        ids=["gemma4", "default"],
    )
    def test_kept_towers_without_tower_lora_leave_kwargs_alone(
        self, runtime: ModelRuntimeConfig
    ) -> None:
        kwargs: dict[str, object] = {}

        apply_multimodal_engine_kwargs(
            kwargs, strip_multimodal_towers=False, runtime=runtime
        )

        assert kwargs == {}
