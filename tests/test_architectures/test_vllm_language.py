# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for language-tower vLLM kwargs on VL checkpoints."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from agilerl.architectures.catalog import FAMILY_RUNTIME_CONFIGS
from agilerl.architectures.gemma4 import (
    GEMMA4_LANGUAGE_ARCHITECTURE,
    gemma4_language_tower_hf_override,
)
from agilerl.architectures.nemotron_h.language_tower import (
    omni_language_tower_hf_override,
)
from agilerl.architectures.qwen3_5 import (
    QWEN3_5_LANGUAGE_ARCHITECTURE,
    QWEN3_5_MODEL_CLASS_OVERRIDES,
    QWEN3_5_MOE_LANGUAGE_ARCHITECTURE,
    qwen3_5_language_tower_hf_override,
)
from agilerl.architectures.runtime import ModelRuntimeConfig
from agilerl.architectures.vllm_language import (
    apply_language_tower_engine_kwargs,
    apply_tower_connector_lora_engine_kwargs,
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


class TestQwen35LanguageForCausalLM:
    @pytest.mark.vllm
    def test_maps_language_prefixes_and_drops_towers(self) -> None:
        pytest.importorskip("vllm")
        from agilerl.architectures.qwen3_5.causal_lm import (
            Qwen3_5MoeLanguageForCausalLM,
        )

        prefixes = Qwen3_5MoeLanguageForCausalLM.hf_to_vllm_mapper.orig_to_new_prefix
        assert prefixes["model.language_model."] == "model."
        assert prefixes["model.visual."] is None
        assert prefixes["mtp."] is None
        assert Qwen3_5MoeLanguageForCausalLM.is_3d_moe_weight is True
        assert Qwen3_5MoeLanguageForCausalLM.is_hybrid is True
        assert callable(Qwen3_5MoeLanguageForCausalLM.get_mamba_state_shape_from_config)


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


class TestGemma4LanguageTowerHfOverride:
    def test_sets_causal_lm_architecture_on_text_config(self) -> None:
        language = SimpleNamespace()
        config = SimpleNamespace(text_config=language)

        resolved = gemma4_language_tower_hf_override(config)

        assert resolved is language
        assert language.architectures == [GEMMA4_LANGUAGE_ARCHITECTURE]

    def test_sets_causal_lm_architecture_on_language_only_config(self) -> None:
        config = SimpleNamespace(model_type="gemma4_text")

        resolved = gemma4_language_tower_hf_override(config)

        assert resolved is config
        assert config.architectures == [GEMMA4_LANGUAGE_ARCHITECTURE]


class TestQwen35LanguageTowerHfOverride:
    def test_sets_moe_architecture_on_text_config(self) -> None:
        language = SimpleNamespace(model_type="qwen3_5_moe_text")
        config = SimpleNamespace(
            model_type="qwen3_5_moe",
            architectures=["Qwen3_5MoeForConditionalGeneration"],
            text_config=language,
        )

        resolved = qwen3_5_language_tower_hf_override(config)

        assert resolved is language
        assert language.architectures == [QWEN3_5_MOE_LANGUAGE_ARCHITECTURE]

    def test_sets_dense_architecture_on_text_config(self) -> None:
        language = SimpleNamespace(model_type="qwen3_5_text")
        config = SimpleNamespace(
            model_type="qwen3_5",
            text_config=language,
        )

        resolved = qwen3_5_language_tower_hf_override(config)

        assert resolved is language
        assert language.architectures == [QWEN3_5_LANGUAGE_ARCHITECTURE]

    def test_sets_moe_architecture_on_language_only_config(self) -> None:
        config = SimpleNamespace(model_type="qwen3_5_moe_text")

        resolved = qwen3_5_language_tower_hf_override(config)

        assert resolved is config
        assert config.architectures == [QWEN3_5_MOE_LANGUAGE_ARCHITECTURE]


class TestApplyTowerConnectorLoraEngineKwargs:
    @pytest.mark.parametrize("strip_multimodal_towers", [False, [], ["audio_tower"]])
    def test_enables_when_family_supports_it_and_towers_load(
        self, strip_multimodal_towers: bool | list[str]
    ) -> None:
        kwargs: dict[str, object] = {}

        apply_tower_connector_lora_engine_kwargs(
            kwargs,
            strip_multimodal_towers,
            runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"],
        )

        assert kwargs == {"enable_tower_connector_lora": True}

    def test_leaves_kwargs_when_serving_language_tower_only(self) -> None:
        kwargs: dict[str, object] = {}

        apply_tower_connector_lora_engine_kwargs(
            kwargs,
            strip_multimodal_towers=True,
            runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"],
        )

        assert kwargs == {}

    def test_leaves_kwargs_when_family_towers_lack_lora(self) -> None:
        kwargs: dict[str, object] = {}

        apply_tower_connector_lora_engine_kwargs(
            kwargs,
            strip_multimodal_towers=False,
            runtime=FAMILY_RUNTIME_CONFIGS["gemma4"],
        )

        assert kwargs == {}


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

    def test_gemma4_strip_sets_language_tower_override(self) -> None:
        kwargs: dict[str, object] = {}

        apply_language_tower_engine_kwargs(
            kwargs,
            strip_multimodal_towers=True,
            runtime=FAMILY_RUNTIME_CONFIGS["gemma4"],
        )

        assert kwargs["hf_overrides"] is gemma4_language_tower_hf_override
        assert "model_class_overrides" not in kwargs

    def test_qwen3_5_strip_sets_language_tower_override(self) -> None:
        kwargs: dict[str, object] = {}

        apply_language_tower_engine_kwargs(
            kwargs,
            strip_multimodal_towers=True,
            runtime=FAMILY_RUNTIME_CONFIGS["qwen3_5_moe"],
        )

        assert kwargs["hf_overrides"] is qwen3_5_language_tower_hf_override
        assert kwargs["model_class_overrides"] == QWEN3_5_MODEL_CLASS_OVERRIDES

    def test_qwen3_5_kept_towers_leave_kwargs_alone(self) -> None:
        kwargs: dict[str, object] = {}

        apply_language_tower_engine_kwargs(
            kwargs,
            strip_multimodal_towers=False,
            runtime=FAMILY_RUNTIME_CONFIGS["qwen3_5_moe"],
        )

        assert kwargs == {}

    def test_gemma4_kept_towers_leave_kwargs_alone(self) -> None:
        kwargs: dict[str, object] = {}

        apply_language_tower_engine_kwargs(
            kwargs,
            strip_multimodal_towers=False,
            runtime=FAMILY_RUNTIME_CONFIGS["gemma4"],
        )

        assert kwargs == {}

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


QWEN3_5_CAUSAL_LM_MODULE = "agilerl.architectures.qwen3_5.causal_lm"


def install_qwen3_5_vllm_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    class WeightsMapper:
        def __init__(self, orig_to_new_prefix: dict[str, str | None]) -> None:
            self.orig_to_new_prefix = orig_to_new_prefix

        def apply(self, weights):
            mapped = []
            for name, data in weights:
                out_name = name
                drop = False
                for prefix, new in self.orig_to_new_prefix.items():
                    if name.startswith(prefix):
                        if new is None:
                            drop = True
                        else:
                            out_name = name.replace(prefix, new, 1)
                        break
                if not drop:
                    mapped.append((out_name, data))
            return mapped

    class Qwen3_5ForCausalLM:
        def load_weights(self, weights) -> set[str]:
            return {name for name, _ in weights}

    class Qwen3_5MoeForCausalLM:
        def load_weights(self, weights) -> set[str]:
            return {name for name, _ in weights}

    class Qwen3_5ForConditionalGeneration:
        @classmethod
        def get_mamba_state_dtype_from_config(cls, vllm_config: object) -> tuple:
            return ()

        @classmethod
        def get_mamba_state_shape_from_config(cls, vllm_config: object) -> tuple:
            return ()

        @classmethod
        def get_mamba_state_copy_func(cls) -> tuple:
            return ()

    class IsHybrid:
        is_hybrid = True

    class SupportsMRoPE:
        supports_mrope = True

    def package(name: str) -> ModuleType:
        mod = ModuleType(name)
        mod.__path__ = []
        return mod

    vllm_mod = package("vllm")
    executor = package("vllm.model_executor")
    models = package("vllm.model_executor.models")
    qwen3_5 = ModuleType("vllm.model_executor.models.qwen3_5")
    interfaces = ModuleType("vllm.model_executor.models.interfaces")
    utils = ModuleType("vllm.model_executor.models.utils")
    qwen3_5.Qwen3_5ForCausalLM = Qwen3_5ForCausalLM
    qwen3_5.Qwen3_5MoeForCausalLM = Qwen3_5MoeForCausalLM
    qwen3_5.Qwen3_5ForConditionalGeneration = Qwen3_5ForConditionalGeneration
    interfaces.IsHybrid = IsHybrid
    interfaces.SupportsMRoPE = SupportsMRoPE
    utils.WeightsMapper = WeightsMapper

    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", executor)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", models)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.qwen3_5",
        qwen3_5,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.interfaces",
        interfaces,
    )
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.utils", utils)
    monkeypatch.delitem(sys.modules, QWEN3_5_CAUSAL_LM_MODULE, raising=False)


class TestQwen35LanguageMapperWithoutVllm:
    def test_class_maps_vl_checkpoint_prefixes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_qwen3_5_vllm_stubs(monkeypatch)

        try:
            mod = importlib.import_module(QWEN3_5_CAUSAL_LM_MODULE)
            mapper = mod.Qwen3_5MoeLanguageForCausalLM.hf_to_vllm_mapper
            assert mapper.orig_to_new_prefix == {
                "model.language_model.": "model.",
                "model.visual.": None,
                "mtp.": None,
            }
            assert (
                mod.Qwen3_5LanguageForCausalLM.hf_to_vllm_mapper
                is mod.Qwen3_5MoeLanguageForCausalLM.hf_to_vllm_mapper
            )
            assert mod.Qwen3_5MoeLanguageForCausalLM.is_3d_moe_weight is True
            assert not hasattr(mod.Qwen3_5LanguageForCausalLM, "is_3d_moe_weight")
            assert mod.Qwen3_5MoeLanguageForCausalLM.is_hybrid is True
            assert callable(
                mod.Qwen3_5MoeLanguageForCausalLM.get_mamba_state_shape_from_config
            )
        finally:
            sys.modules.pop(QWEN3_5_CAUSAL_LM_MODULE, None)

    def test_language_towers_compute_text_mrope_positions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_qwen3_5_vllm_stubs(monkeypatch)

        try:
            mod = importlib.import_module(QWEN3_5_CAUSAL_LM_MODULE)
            towers = (
                mod.Qwen3_5LanguageForCausalLM,
                mod.Qwen3_5MoeLanguageForCausalLM,
            )

            for tower in towers:
                positions, delta = tower().get_mrope_input_positions([1, 2, 3, 4], [])

                assert tower.supports_mrope is True
                assert positions.tolist() == [[0, 1, 2, 3]] * 3
                assert delta == 0

            with pytest.raises(ValueError, match="text-only"):
                mod.Qwen3_5MoeLanguageForCausalLM().get_mrope_input_positions(
                    [1, 2], [object()]
                )
        finally:
            sys.modules.pop(QWEN3_5_CAUSAL_LM_MODULE, None)

    def test_language_towers_load_mapped_vl_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_qwen3_5_vllm_stubs(monkeypatch)

        try:
            mod = importlib.import_module(QWEN3_5_CAUSAL_LM_MODULE)
            weights = (
                ("model.language_model.embed", object()),
                ("model.visual.patch", object()),
                ("mtp.head", object()),
            )

            for tower in (
                mod.Qwen3_5LanguageForCausalLM,
                mod.Qwen3_5MoeLanguageForCausalLM,
            ):
                assert tower().load_weights(weights) == {"model.embed"}
        finally:
            sys.modules.pop(QWEN3_5_CAUSAL_LM_MODULE, None)


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
