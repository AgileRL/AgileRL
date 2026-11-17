# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for multimodal vLLM engine kwargs on VL checkpoints."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agilerl.architectures.vllm_language import (
    apply_multimodal_engine_kwargs,
    nested_language_config,
)


class TestQwen35ConditionalGenerationLoraNames:
    @pytest.mark.vllm
    def test_vl_trainer_lora_names_parse_to_stock_language_layers(self) -> None:
        pytest.importorskip("vllm")
        from vllm.lora.utils import parse_fine_tuned_lora_name
        from vllm.model_executor.models.qwen3_5 import (
            Qwen3_5ForConditionalGeneration,
        )

        mapper = Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper.get_rename_mapper()

        module, _ = parse_fine_tuned_lora_name(
            "base_model.model.model.language_model.layers.0.self_attn.q_proj"
            ".lora_A.weight",
            mapper,
        )

        assert module == "language_model.model.layers.0.self_attn.q_proj"


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
    def test_strip_all_serves_language_model_only(self) -> None:
        kwargs: dict[str, object] = {}

        apply_multimodal_engine_kwargs(kwargs, strip_multimodal_towers=True)

        assert kwargs == {"language_model_only": True}

    def test_kept_towers_leave_kwargs_alone(self) -> None:
        kwargs: dict[str, object] = {}

        apply_multimodal_engine_kwargs(kwargs, strip_multimodal_towers=False)

        assert kwargs == {}
