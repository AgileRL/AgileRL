# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for model-type trainer and vLLM runtime configs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.configuration_utils import PretrainedConfig
from transformers.models.nemotron_h_omni.configuration_nemotron_h_omni import (
    NemotronH_Omni_Reasoning_V3_Config,
)

from agilerl.architectures import family_tensor_parallel_plans
from agilerl.architectures.catalog import (
    FAMILY_RUNTIME_CONFIGS,
    family_runtime,
    nemotron_h_omni_vllm_hf_override,
    pretrained_model_type,
    trainer_model_class,
)
from agilerl.architectures.nemotron_h.mamba import install_mamba_patches
from agilerl.architectures.nemotron_h.tensor_parallel import (
    NEMOTRON_H_TENSOR_PARALLEL_PLAN,
)
from agilerl.architectures.runtime import ModelRuntimeConfig, VllmRuntimeConfig
from agilerl.arena.memory.manifest import FAMILY_MAX_NUM_BATCHED_TOKENS
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.llm_envs.tool_parsers import TOOL_PARSERS

NEMOTRON_VLLM_KWARGS = {
    "mamba_cache_mode": "align",
    "max_num_batched_tokens": 8192,
    "reasoning_parser": "nemotron_v3",
    "enable_prefix_caching": True,
    "prefix_cache_retention_interval": None,
}

SWA_MODEL_TYPES = ("gemma3", "gemma3_text", "gemma4", "gemma4_text")


def stub_config_model_type(monkeypatch: pytest.MonkeyPatch, model_type: str) -> None:
    @classmethod
    def fake_get_config_dict(
        cls, pretrained_model_name_or_path: str, **kwargs: object
    ) -> tuple[dict[str, object], dict[str, object]]:
        return ({"model_type": model_type}, {})

    monkeypatch.setattr(
        "agilerl.architectures.catalog.PretrainedConfig.get_config_dict",
        fake_get_config_dict,
    )


class TestFamilyRuntimeConfigs:
    def test_catalog_keys(self) -> None:
        assert set(FAMILY_RUNTIME_CONFIGS) == {
            "qwen2",
            "qwen3",
            "qwen3_moe",
            "granitemoe",
            "granitemoehybrid",
            "nemotron_h",
            "nemotron_h_omni",
            "gemma3",
            "gemma3_text",
            "gemma4",
            "gemma4_text",
            "gpt_oss",
            "qwen3_5",
            "qwen3_5_text",
            "qwen3_5_moe",
            "qwen3_5_moe_text",
        }

    def test_nemotron_h_omni_keeps_nemotron_h_runtime_and_sets_the_vllm_architecture(
        self,
    ) -> None:
        base = FAMILY_RUNTIME_CONFIGS["nemotron_h"]
        omni = FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"]
        assert omni.vllm.mamba_cache_mode == base.vllm.mamba_cache_mode
        assert omni.vllm.hf_overrides is nemotron_h_omni_vllm_hf_override
        assert base.vllm.hf_overrides is None
        assert omni.patch.install is install_mamba_patches
        assert {"q_proj", "fc1", "linear_1"} <= omni.tower_connector_lora_targets
        assert omni.tower_connector_lora_targets.isdisjoint(
            {"in_proj", "up_proj", "down_proj", "out_proj"}
        )
        assert base.tower_connector_lora_targets == frozenset()

    def test_gemma4_uses_the_swa_trainer(self) -> None:
        gemma4 = FAMILY_RUNTIME_CONFIGS["gemma4"]
        gemma3 = FAMILY_RUNTIME_CONFIGS["gemma3"]
        assert gemma4.trainer == gemma3.trainer

    def test_every_arena_tower_lora_module_is_served(self) -> None:
        unserved = {
            hub_id: sorted(
                entry.tower_lora_modules
                - FAMILY_RUNTIME_CONFIGS[
                    entry.config["model_type"]
                ].tower_connector_lora_targets
            )
            for hub_id, entry in SUPPORTED_MODEL_INFO.items()
        }

        assert {hub_id: names for hub_id, names in unserved.items() if names} == {}

    @pytest.mark.parametrize(
        "model_type",
        ["qwen3_5", "qwen3_5_moe", "gemma3", "gemma4", "nemotron_h_omni"],
    )
    def test_vl_families_train_their_vl_class(self, model_type: str) -> None:
        trainer = FAMILY_RUNTIME_CONFIGS[model_type].trainer

        assert trainer.model_class is AutoModelForImageTextToText

    @pytest.mark.parametrize(
        "model_type",
        sorted(
            set(FAMILY_RUNTIME_CONFIGS)
            - {"qwen3_5", "qwen3_5_moe", "gemma3", "gemma4", "nemotron_h_omni"}
        ),
    )
    def test_text_families_train_a_causal_lm(self, model_type: str) -> None:
        trainer = FAMILY_RUNTIME_CONFIGS[model_type].trainer

        assert trainer.model_class is AutoModelForCausalLM

    def test_nemotron_h_lookup(self) -> None:
        config = FAMILY_RUNTIME_CONFIGS["nemotron_h"]
        assert config.vllm.engine_kwargs() == NEMOTRON_VLLM_KWARGS
        assert config.trainer.load_kwargs() == {}
        assert config.trainer.attn_implementation == "flash_attention_2"
        assert config.patch.install is install_mamba_patches

    def test_no_family_loads_checkpoint_code(self) -> None:
        omni = FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"]

        assert omni.trainer.load_kwargs() == {"trust_remote_code": False}
        assert all(
            runtime.trainer.trust_remote_code is not True
            and runtime.vllm.trust_remote_code is not True
            for runtime in FAMILY_RUNTIME_CONFIGS.values()
        )

    def test_nemotron_h_enables_prefix_caching(self) -> None:
        assert FAMILY_RUNTIME_CONFIGS["nemotron_h"].vllm.enable_prefix_caching is True

    @pytest.mark.parametrize("model_type", SWA_MODEL_TYPES)
    def test_swa_lookup(self, model_type: str) -> None:
        assert (
            FAMILY_RUNTIME_CONFIGS[model_type].trainer.attn_implementation
            == "flex_attention"
        )

    def test_gpt_oss_lookup(self) -> None:
        assert (
            FAMILY_RUNTIME_CONFIGS["gpt_oss"].trainer.attn_implementation
            == "flex_attention"
        )

    def test_qwen3_5_lookup(self) -> None:
        runtime = FAMILY_RUNTIME_CONFIGS["qwen3_5"]
        assert runtime.vllm.mamba_cache_mode == "align"
        assert runtime.vllm.enable_prefix_caching is True
        assert runtime.tower_connector_lora_targets == {
            "qkv",
            "linear_fc1",
            "linear_fc2",
        }

    @pytest.mark.parametrize(
        "model_type", ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"]
    )
    def test_qwen3_5_families_name_a_reasoning_parser(self, model_type: str) -> None:
        kwargs = FAMILY_RUNTIME_CONFIGS[model_type].vllm.engine_kwargs()

        assert kwargs["reasoning_parser"] == "qwen3"

    def test_every_supported_model_names_a_registered_tool_call_parser(
        self,
    ) -> None:
        model_types = {
            info.config["model_type"] for info in SUPPORTED_MODEL_INFO.values()
        }

        parsers = {
            model_type: FAMILY_RUNTIME_CONFIGS[model_type].tool_call_parser
            for model_type in model_types
        }

        assert all(name in TOOL_PARSERS for name in parsers.values()), parsers

    def test_catalog_excludes_gemma_and_gemma2(self) -> None:
        assert "gemma" not in FAMILY_RUNTIME_CONFIGS
        assert "gemma2" not in FAMILY_RUNTIME_CONFIGS

    def test_nemotron_has_mamba_patches(self) -> None:
        patch = FAMILY_RUNTIME_CONFIGS["nemotron_h"].patch
        assert patch.install is install_mamba_patches
        mamba = patch.mamba
        assert mamba is not None
        assert (
            mamba.mixer
            == "transformers.models.nemotron_h.modeling_nemotron_h.NemotronHMamba2Mixer"
        )
        assert mamba.fused_path is True
        assert mamba.stream_ordering is True


class TestVllmRuntimeConfigEngineKwargs:
    def test_dense_mamba_prefix_cache_retains_every_state(self) -> None:
        config = VllmRuntimeConfig(
            mamba_cache_mode="align", dense_mamba_prefix_cache=True
        )

        kwargs = config.engine_kwargs()

        assert kwargs == {
            "mamba_cache_mode": "align",
            "prefix_cache_retention_interval": None,
        }

    def test_unset_fields_add_no_kwargs(self) -> None:
        assert VllmRuntimeConfig(reasoning_parser="qwen3").engine_kwargs() == {
            "reasoning_parser": "qwen3"
        }

    @pytest.mark.parametrize("model_type", ["nemotron_h", "qwen3_5"])
    def test_hybrid_mamba_families_cache_densely(self, model_type: str) -> None:
        kwargs = FAMILY_RUNTIME_CONFIGS[model_type].vllm.engine_kwargs()

        assert kwargs["prefix_cache_retention_interval"] is None


class TestFamilyMaxNumBatchedTokens:
    def test_arena_estimator_matches_the_catalog(self) -> None:
        catalog = {
            family: runtime.vllm.max_num_batched_tokens
            for family, runtime in FAMILY_RUNTIME_CONFIGS.items()
            if runtime.vllm.max_num_batched_tokens is not None
        }

        assert catalog == FAMILY_MAX_NUM_BATCHED_TOKENS


class TestFamilyTensorParallelPlans:
    def test_resolves_shared_nemotron_h_plan_once(self) -> None:
        plans = family_tensor_parallel_plans()

        assert plans == [NEMOTRON_H_TENSOR_PARALLEL_PLAN]

    def test_missing_plan_module_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(
            FAMILY_RUNTIME_CONFIGS,
            "missing_family",
            ModelRuntimeConfig(tensor_parallel_plan="agilerl.no_such_module:PLAN"),
        )

        with pytest.raises(ModuleNotFoundError, match=r"agilerl\.no_such_module"):
            family_tensor_parallel_plans()


class TestFamilyRuntime:
    @pytest.mark.parametrize("model_type", ["nemotron_h", "nemotron_h_omni"])
    def test_nemotron_hub_id_uses_catalog(
        self, monkeypatch: pytest.MonkeyPatch, model_type: str
    ) -> None:
        stub_config_model_type(monkeypatch, model_type)
        config = family_runtime("nvidia/nemotron")
        assert config is FAMILY_RUNTIME_CONFIGS[model_type]

    @pytest.mark.parametrize("model_type", SWA_MODEL_TYPES)
    def test_swa_hub_id_uses_catalog(
        self, monkeypatch: pytest.MonkeyPatch, model_type: str
    ) -> None:
        stub_config_model_type(monkeypatch, model_type)
        config = family_runtime("google/gemma")
        assert config.trainer == FAMILY_RUNTIME_CONFIGS[model_type].trainer

    def test_gpt_oss_hub_id_uses_catalog(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_config_model_type(monkeypatch, "gpt_oss")
        config = family_runtime("openai/gpt-oss-20b")
        assert config.trainer == FAMILY_RUNTIME_CONFIGS["gpt_oss"].trainer

    @pytest.mark.parametrize("model_type", ["gemma", "gemma2", "phi3", "llama"])
    def test_unknown_hub_type_returns_empty_defaults(
        self, monkeypatch: pytest.MonkeyPatch, model_type: str
    ) -> None:
        stub_config_model_type(monkeypatch, model_type)
        config = family_runtime("some/model")
        assert config.vllm.engine_kwargs() == {}
        assert config.trainer.load_kwargs() == {}
        assert config.trainer.model_class is AutoModelForCausalLM
        assert config.patch.install is None

    def test_missing_config_json_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_missing(*args: object, **kwargs: object) -> None:
            msg = "missing config"
            raise OSError(msg)

        monkeypatch.setattr(
            "agilerl.architectures.catalog.PretrainedConfig.get_config_dict",
            raise_missing,
        )
        with pytest.raises(OSError, match="missing config"):
            family_runtime("nvidia/unlisted-model")

    @pytest.mark.parametrize(
        ("hub_id", "model_type"),
        [
            ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "nemotron_h"),
            ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "nemotron_h"),
            ("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", "nemotron_h_omni"),
        ],
    )
    def test_supported_id_resolves_from_bundle_without_hub(
        self, monkeypatch: pytest.MonkeyPatch, hub_id: str, model_type: str
    ) -> None:
        def raise_if_called(*args: object, **kwargs: object) -> None:
            msg = "hub lookup"
            raise AssertionError(msg)

        monkeypatch.setattr(
            "agilerl.architectures.catalog.PretrainedConfig.get_config_dict",
            raise_if_called,
        )

        config = family_runtime(hub_id)

        assert config is FAMILY_RUNTIME_CONFIGS[model_type]


class TestPretrainedModelType:
    def test_reads_config_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        @classmethod
        def fake_get_config_dict(
            cls, pretrained_model_name_or_path: str, **kwargs: object
        ) -> tuple[dict[str, object], dict[str, object]]:
            seen.update(kwargs)
            return ({"model_type": "nemotron_h"}, {})

        monkeypatch.setattr(
            "agilerl.architectures.catalog.PretrainedConfig.get_config_dict",
            fake_get_config_dict,
        )
        assert pretrained_model_type("nvidia/unlisted-model") == "nemotron_h"
        assert "trust_remote_code" not in seen

    def test_missing_config_json_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_missing(*args: object, **kwargs: object) -> None:
            msg = "missing config"
            raise OSError(msg)

        monkeypatch.setattr(
            "agilerl.architectures.catalog.PretrainedConfig.get_config_dict",
            raise_missing,
        )
        with pytest.raises(OSError, match="missing config"):
            pretrained_model_type("nvidia/unlisted-model")

    def test_supported_id_reads_bundled_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_if_called(*args: object, **kwargs: object) -> None:
            msg = "hub lookup"
            raise AssertionError(msg)

        monkeypatch.setattr(
            "agilerl.architectures.catalog.PretrainedConfig.get_config_dict",
            raise_if_called,
        )

        assert pretrained_model_type("google/gemma-4-E4B-it") == "gemma4"


class TestRuntimeConfigsForbidExtra:
    def test_unknown_fields_are_rejected(self) -> None:
        from agilerl.architectures.runtime import (
            MambaPatchConfig,
            ModelRuntimeConfig,
            PatchRuntimeConfig,
            TrainerRuntimeConfig,
            VllmRuntimeConfig,
        )

        cases = (
            (VllmRuntimeConfig, {}),
            (TrainerRuntimeConfig, {}),
            (MambaPatchConfig, {"mixer": "agilerl.architectures.nemotron_h.mamba"}),
            (PatchRuntimeConfig, {}),
            (ModelRuntimeConfig, {}),
        )
        for cls, payload in cases:
            with pytest.raises(ValidationError, match="extra_forbidden"):
                cls.model_validate({**payload, "__unknown__": True})


class TestTrainerModelClass:
    @pytest.mark.parametrize(
        ("hub_id", "expected"),
        [
            ("Qwen/Qwen3.5-2B", AutoModelForImageTextToText),
            ("Qwen/Qwen3.6-35B-A3B", AutoModelForImageTextToText),
            ("google/gemma-4-E4B-it", AutoModelForImageTextToText),
            (
                "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
                AutoModelForImageTextToText,
            ),
            ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", AutoModelForCausalLM),
            ("openai/gpt-oss-20b", AutoModelForCausalLM),
        ],
    )
    def test_supported_ids_resolve_their_trainer_class(
        self, hub_id: str, expected: type
    ) -> None:
        assert trainer_model_class(hub_id) is expected

    def test_unknown_family_defaults_to_causal_lm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_config_model_type(monkeypatch, "llama")

        assert trainer_model_class("some/model") is AutoModelForCausalLM


class TestNemotronHOmniVllmHfOverride:
    def test_native_config_gets_the_lora_architecture_and_vit_hidden_size(
        self,
    ) -> None:
        # Arrange
        config = NemotronH_Omni_Reasoning_V3_Config(vision_hidden_size=1024)

        # Act
        overridden = nemotron_h_omni_vllm_hf_override(config)

        # Assert
        assert overridden is config
        assert config.architectures == ["NemotronH_Super_Omni_Reasoning_V3"]
        assert config.vit_hidden_size == 1024

    def test_probe_config_without_a_vision_size_gets_only_the_architecture(
        self,
    ) -> None:
        config = PretrainedConfig()

        nemotron_h_omni_vllm_hf_override(config)

        assert config.architectures == ["NemotronH_Super_Omni_Reasoning_V3"]
        assert not hasattr(config, "vit_hidden_size")
