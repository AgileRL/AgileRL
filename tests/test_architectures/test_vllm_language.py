# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for multimodal vLLM engine kwargs on VL checkpoints."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agilerl.architectures.catalog import FAMILY_RUNTIME_CONFIGS
from agilerl.architectures.vllm_language import (
    apply_multimodal_engine_kwargs,
    apply_tower_connector_lora_engine_kwargs,
    nested_language_config,
    unbuilt_tower_modules,
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


class TestApplyTowerConnectorLoraEngineKwargs:
    def test_enables_when_tower_targets_are_named(self) -> None:
        kwargs: dict[str, object] = {}

        apply_tower_connector_lora_engine_kwargs(
            kwargs,
            strip_multimodal_towers=False,
            runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"],
            tower_target_modules=["q_proj", "linear_1"],
        )

        assert kwargs == {"enable_tower_connector_lora": True}

    @pytest.mark.parametrize(
        ("family", "targets"),
        [
            ("qwen3_5", ["qkv", "linear_fc1", "linear_fc2"]),
            ("qwen3_5_moe", ["qkv", "linear_fc1", "linear_fc2"]),
            ("gemma4", ["linear", "input_proj", "embedding_projection"]),
        ],
    )
    def test_enables_vision_tower_targets_of_vl_families(
        self, family: str, targets: list[str]
    ) -> None:
        kwargs: dict[str, object] = {}

        apply_tower_connector_lora_engine_kwargs(
            kwargs,
            strip_multimodal_towers=False,
            runtime=FAMILY_RUNTIME_CONFIGS[family],
            tower_target_modules=targets,
        )

        assert kwargs == {"enable_tower_connector_lora": True}

    @pytest.mark.parametrize("tower_target_modules", [None, []])
    def test_leaves_kwargs_without_tower_targets(
        self, tower_target_modules: list[str] | None
    ) -> None:
        kwargs: dict[str, object] = {}

        apply_tower_connector_lora_engine_kwargs(
            kwargs,
            strip_multimodal_towers=False,
            runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"],
            tower_target_modules=tower_target_modules,
        )

        assert kwargs == {}

    def test_stripped_towers_reject_tower_targets(self) -> None:
        with pytest.raises(ValueError, match="the engine strips them"):
            apply_tower_connector_lora_engine_kwargs(
                {},
                strip_multimodal_towers=True,
                runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"],
                tower_target_modules=["q_proj"],
            )

    @pytest.mark.parametrize(
        ("family", "targets"),
        [
            ("gemma4", ["q_proj"]),
            ("qwen3_5", ["proj"]),
            ("nemotron_h_omni", ["in_proj"]),
        ],
    )
    def test_rejects_targets_vllm_cannot_serve(
        self, family: str, targets: list[str]
    ) -> None:
        with pytest.raises(ValueError, match="cannot serve tower LoRA"):
            apply_tower_connector_lora_engine_kwargs(
                {},
                strip_multimodal_towers=False,
                runtime=FAMILY_RUNTIME_CONFIGS[family],
                tower_target_modules=targets,
            )


class TestUnbuiltTowerModules:
    def test_strip_all_drops_every_tower(self) -> None:
        unbuilt = unbuilt_tower_modules(True, FAMILY_RUNTIME_CONFIGS["gemma4"])

        assert unbuilt == ["audio_tower", "embed_audio", "embed_vision", "vision_tower"]

    def test_served_towers_are_all_built(self) -> None:
        unbuilt = unbuilt_tower_modules(False, FAMILY_RUNTIME_CONFIGS["gemma4"])

        assert unbuilt == []


class TestApplyMultimodalEngineKwargs:
    def test_strip_all_serves_language_model_only(self) -> None:
        kwargs: dict[str, object] = {}

        apply_multimodal_engine_kwargs(kwargs, strip_multimodal_towers=True)

        assert kwargs == {"language_model_only": True}

    def test_kept_towers_leave_kwargs_alone(self) -> None:
        kwargs: dict[str, object] = {}

        apply_multimodal_engine_kwargs(kwargs, strip_multimodal_towers=False)

        assert kwargs == {}
