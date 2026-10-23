# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for multimodal vLLM engine kwargs on VL checkpoints."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from agilerl.architectures.catalog import FAMILY_RUNTIME_CONFIGS
from agilerl.architectures.qwen3_5 import (
    QWEN3_5_LANGUAGE_ARCHITECTURE,
    QWEN3_5_MODEL_CLASS_OVERRIDES,
    QWEN3_5_MOE_LANGUAGE_ARCHITECTURE,
    qwen3_5_language_tower_hf_override,
)
from agilerl.architectures.vllm_language import (
    apply_multimodal_engine_kwargs,
    apply_tower_connector_lora_engine_kwargs,
    nested_language_config,
)


class TestQwen35LanguageForCausalLM:
    @pytest.mark.vllm
    def test_maps_language_prefixes_and_drops_towers(self) -> None:
        pytest.importorskip("vllm")
        from agilerl.architectures.qwen3_5.causal_lm import (
            Qwen3_5MoeLanguageForCausalLM,
        )

        prefixes = Qwen3_5MoeLanguageForCausalLM.hf_to_vllm_mapper.orig_to_new_prefix
        assert prefixes["model.language_model."] == "model."
        assert prefixes["language_model."] == "model."
        assert prefixes["model.visual."] is None
        assert prefixes["mtp."] is None
        assert Qwen3_5MoeLanguageForCausalLM.is_3d_moe_weight is True
        assert Qwen3_5MoeLanguageForCausalLM.is_hybrid is True
        assert callable(Qwen3_5MoeLanguageForCausalLM.get_mamba_state_shape_from_config)

    @pytest.mark.vllm
    def test_lora_names_parse_to_tower_layers(self) -> None:
        pytest.importorskip("vllm")
        from vllm.lora.utils import parse_fine_tuned_lora_name

        from agilerl.architectures.qwen3_5.causal_lm import (
            Qwen3_5LanguageForCausalLM,
        )

        mapper = Qwen3_5LanguageForCausalLM.hf_to_vllm_mapper.get_unstacked_mapper()

        module, _ = parse_fine_tuned_lora_name(
            "base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
            mapper,
        )

        assert module == "model.layers.0.self_attn.q_proj"


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
    def test_enables_when_targets_reach_kept_towers(
        self, strip_multimodal_towers: bool | list[str]
    ) -> None:
        kwargs: dict[str, object] = {}

        apply_tower_connector_lora_engine_kwargs(
            kwargs,
            strip_multimodal_towers,
            runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"],
            lora_target_modules=["q_proj", "query", "linear1"],
        )

        assert kwargs == {"enable_tower_connector_lora": True}

    @pytest.mark.parametrize("lora_target_modules", ["all-linear", r".*\.fc1", None])
    def test_enables_for_target_specs_without_names(
        self, lora_target_modules: str | None
    ) -> None:
        kwargs: dict[str, object] = {}

        apply_tower_connector_lora_engine_kwargs(
            kwargs,
            strip_multimodal_towers=False,
            runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"],
            lora_target_modules=lora_target_modules,
        )

        assert kwargs == {"enable_tower_connector_lora": True}

    def test_leaves_kwargs_when_targets_are_language_only(self) -> None:
        kwargs: dict[str, object] = {}

        apply_tower_connector_lora_engine_kwargs(
            kwargs,
            strip_multimodal_towers=False,
            runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"],
            lora_target_modules={"q_proj", "k_proj", "in_proj", "up_proj"},
        )

        assert kwargs == {}

    def test_leaves_kwargs_when_serving_language_tower_only(self) -> None:
        kwargs: dict[str, object] = {}

        apply_tower_connector_lora_engine_kwargs(
            kwargs,
            strip_multimodal_towers=True,
            runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"],
            lora_target_modules="all-linear",
        )

        assert kwargs == {}

    def test_leaves_kwargs_when_family_towers_lack_lora(self) -> None:
        kwargs: dict[str, object] = {}

        apply_tower_connector_lora_engine_kwargs(
            kwargs,
            strip_multimodal_towers=False,
            runtime=FAMILY_RUNTIME_CONFIGS["gemma4"],
            lora_target_modules="all-linear",
        )

        assert kwargs == {}


class TestApplyMultimodalEngineKwargs:
    @pytest.mark.parametrize("family", ["nemotron_h_omni", "gemma4", "nemotron_h"])
    def test_strip_all_serves_language_model_only(self, family: str) -> None:
        kwargs: dict[str, object] = {}

        apply_multimodal_engine_kwargs(
            kwargs,
            strip_multimodal_towers=True,
            runtime=FAMILY_RUNTIME_CONFIGS[family],
        )

        assert kwargs == {"language_model_only": True}

    def test_qwen3_5_strip_sets_language_tower_override(self) -> None:
        kwargs: dict[str, object] = {}

        apply_multimodal_engine_kwargs(
            kwargs,
            strip_multimodal_towers=True,
            runtime=FAMILY_RUNTIME_CONFIGS["qwen3_5_moe"],
        )

        assert kwargs["hf_overrides"] is qwen3_5_language_tower_hf_override
        assert kwargs["model_class_overrides"] == QWEN3_5_MODEL_CLASS_OVERRIDES
        assert "language_model_only" not in kwargs

    def test_qwen3_5_kept_towers_leave_kwargs_alone(self) -> None:
        kwargs: dict[str, object] = {}

        apply_multimodal_engine_kwargs(
            kwargs,
            strip_multimodal_towers=False,
            runtime=FAMILY_RUNTIME_CONFIGS["qwen3_5_moe"],
        )

        assert kwargs == {}

    def test_modality_list_zeroes_those_limits(self) -> None:
        kwargs: dict[str, object] = {"limit_mm_per_prompt": {"image": 4}}

        apply_multimodal_engine_kwargs(
            kwargs,
            strip_multimodal_towers=["audio"],
            runtime=FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"],
        )

        assert kwargs == {"limit_mm_per_prompt": {"image": 4, "audio": 0}}

    def test_kept_towers_leave_kwargs_alone(self) -> None:
        kwargs: dict[str, object] = {}

        apply_multimodal_engine_kwargs(
            kwargs,
            strip_multimodal_towers=False,
            runtime=FAMILY_RUNTIME_CONFIGS["gemma4"],
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
                "language_model.": "model.",
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
