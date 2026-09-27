# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for language-tower vLLM kwargs on VL checkpoints."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from agilerl.architectures.catalog import FAMILY_RUNTIME_CONFIGS
from agilerl.architectures.nemotron_h.language_tower import (
    omni_language_tower_hf_override,
)
from agilerl.architectures.runtime import ModelRuntimeConfig
from agilerl.architectures.vllm_language import (
    apply_language_tower_engine_kwargs,
    nested_language_config,
)


class TestNemotronHOmniLanguageForCausalLM:
    @pytest.mark.vllm
    def test_maps_language_prefixes_and_drops_towers(self) -> None:
        pytest.importorskip("vllm")
        from agilerl.architectures.nemotron_h.omni_language import (
            NemotronHOmniLanguageForCausalLM,
        )

        assert NemotronHOmniLanguageForCausalLM.is_3d_moe_weight is True
        prefixes = NemotronHOmniLanguageForCausalLM.hf_to_vllm_mapper.orig_to_new_prefix
        assert prefixes["language_model.backbone."] == "model."
        assert prefixes["language_model.lm_head."] == "lm_head."
        assert prefixes["vision_model."] is None
        assert prefixes["mlp1."] is None


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


class TestOmniLanguageTowerHfOverride:
    def test_omni_text_config_rewrites_architectures(self) -> None:
        language = SimpleNamespace(architectures=["NemotronHForCausalLM"])
        omni = SimpleNamespace(
            architectures=["NemotronH_Omni_Reasoning_V3"],
            text_config=language,
        )

        resolved = omni_language_tower_hf_override(omni)

        assert resolved is language
        assert language.architectures == ["NemotronHOmniLanguageForCausalLM"]

    def test_omni_llm_config_rewrites_architectures(self) -> None:
        language = SimpleNamespace(architectures=["NemotronHForCausalLM"])
        omni = SimpleNamespace(
            architectures=["NemotronH_Omni_Reasoning_V3"],
            llm_config=language,
        )

        resolved = omni_language_tower_hf_override(omni)

        assert resolved is language
        assert language.architectures == ["NemotronHOmniLanguageForCausalLM"]

    def test_omni_raises_without_architectures(self) -> None:
        omni = SimpleNamespace(text_config=SimpleNamespace())

        with pytest.raises(TypeError, match="architectures"):
            omni_language_tower_hf_override(omni)


class TestApplyLanguageTowerEngineKwargs:
    def test_omni_family_sets_overrides_when_stripping(self) -> None:
        kwargs: dict[str, object] = {}

        apply_language_tower_engine_kwargs(
            kwargs,
            strip_multimodal_towers=True,
            runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"],
        )

        assert kwargs["hf_overrides"] is omni_language_tower_hf_override
        assert kwargs["model_class_overrides"] == {
            "NemotronHOmniLanguageForCausalLM": (
                "agilerl.architectures.nemotron_h.omni_language:"
                "NemotronHOmniLanguageForCausalLM"
            ),
        }

    def test_generic_strip_peels_nested_config_only(self) -> None:
        kwargs: dict[str, object] = {}

        apply_language_tower_engine_kwargs(
            kwargs,
            strip_multimodal_towers=True,
            runtime=ModelRuntimeConfig(),
        )

        assert kwargs["hf_overrides"] is nested_language_config
        assert "model_class_overrides" not in kwargs

    def test_nemotron_h_strip_does_not_register_omni_class(self) -> None:
        kwargs: dict[str, object] = {}

        apply_language_tower_engine_kwargs(
            kwargs,
            strip_multimodal_towers=True,
            runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h"],
        )

        assert kwargs["hf_overrides"] is nested_language_config
        assert "model_class_overrides" not in kwargs

    def test_omni_family_sets_vision_override_when_towers_kept(self) -> None:
        kwargs: dict[str, object] = {}

        apply_language_tower_engine_kwargs(
            kwargs,
            strip_multimodal_towers=False,
            runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"],
        )

        assert kwargs["hf_overrides"] == {
            "architectures": ["NemotronH_Super_Omni_Reasoning_V3"],
        }
        assert "model_class_overrides" not in kwargs

    def test_named_tower_list_keeps_omni_vision_override(self) -> None:
        kwargs: dict[str, object] = {}

        apply_language_tower_engine_kwargs(
            kwargs,
            strip_multimodal_towers=["audio_tower"],
            runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"],
        )

        assert kwargs["hf_overrides"] == {
            "architectures": ["NemotronH_Super_Omni_Reasoning_V3"],
        }
        assert "model_class_overrides" not in kwargs

    def test_kept_towers_without_family_override_leaves_kwargs_alone(self) -> None:
        kwargs: dict[str, object] = {}

        apply_language_tower_engine_kwargs(
            kwargs,
            strip_multimodal_towers=False,
            runtime=ModelRuntimeConfig(),
        )

        assert kwargs == {}


class TestOmniLanguageMapperWithoutVllm:
    def test_class_maps_omni_checkpoint_prefixes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module_name = "agilerl.architectures.nemotron_h.omni_language"

        class WeightsMapper:
            def __init__(
                self,
                orig_to_new_substr: dict[str, str],
                orig_to_new_prefix: dict[str, str | None],
            ) -> None:
                self.orig_to_new_substr = orig_to_new_substr
                self.orig_to_new_prefix = orig_to_new_prefix

        class NemotronHForCausalLM:
            pass

        def package(name: str) -> ModuleType:
            mod = ModuleType(name)
            mod.__path__ = []
            return mod

        vllm_mod = package("vllm")
        executor = package("vllm.model_executor")
        models = package("vllm.model_executor.models")
        nemotron_h = ModuleType("vllm.model_executor.models.nemotron_h")
        utils = ModuleType("vllm.model_executor.models.utils")
        nemotron_h.NemotronHForCausalLM = NemotronHForCausalLM
        utils.WeightsMapper = WeightsMapper

        monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
        monkeypatch.setitem(sys.modules, "vllm.model_executor", executor)
        monkeypatch.setitem(sys.modules, "vllm.model_executor.models", models)
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.models.nemotron_h",
            nemotron_h,
        )
        monkeypatch.setitem(sys.modules, "vllm.model_executor.models.utils", utils)
        monkeypatch.delitem(sys.modules, module_name, raising=False)

        try:
            mod = importlib.import_module(module_name)
            cls = mod.NemotronHOmniLanguageForCausalLM
            assert cls.is_3d_moe_weight is True
            mapper = cls.hf_to_vllm_mapper
            assert mapper.orig_to_new_substr == {
                "A_log": "A",
                "embeddings": "embed_tokens",
            }
            assert mapper.orig_to_new_prefix == {
                "language_model.mtp.": None,
                "language_model.backbone.": "model.",
                "language_model.lm_head.": "lm_head.",
                "vision_model.": None,
                "vision_projector.": None,
                "mlp1.": None,
            }
        finally:
            sys.modules.pop(module_name, None)
