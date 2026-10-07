# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for model-type trainer and vLLM runtime configs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agilerl.architectures import family_tensor_parallel_plans
from agilerl.architectures.catalog import (
    FAMILY_RUNTIME_CONFIGS,
    family_runtime,
    pretrained_model_type,
)
from agilerl.architectures.gemma4 import gemma4_language_tower_hf_override
from agilerl.architectures.nemotron_h.language_tower import (
    omni_language_tower_hf_override,
)
from agilerl.architectures.nemotron_h.mamba import install_mamba_patches
from agilerl.architectures.nemotron_h.tensor_parallel import (
    NEMOTRON_H_TENSOR_PARALLEL_PLAN,
)
from agilerl.architectures.runtime import ModelRuntimeConfig

NEMOTRON_VLLM_KWARGS = {
    "mamba_cache_mode": "align",
    "max_num_batched_tokens": 8192,
    "reasoning_parser": "nemotron_v3",
    "enable_prefix_caching": True,
    "trust_remote_code": True,
}

NEMOTRON_TRAINER_KWARGS = {"attn_implementation": "flash_attention_2"}
NEMOTRON_OMNI_TRAINER_KWARGS = {
    "attn_implementation": "flash_attention_2",
    "trust_remote_code": True,
}
GEMMA_TRAINER_KWARGS = {"attn_implementation": "flex_attention"}
EMPTY_TRAINER_KWARGS: dict[str, object] = {}
FLEX_TRAINER_KWARGS = GEMMA_TRAINER_KWARGS

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

    def test_nemotron_h_omni_keeps_nemotron_h_runtime_and_adds_language_tower(
        self,
    ) -> None:
        base = FAMILY_RUNTIME_CONFIGS["nemotron_h"]
        omni = FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"]
        assert omni.vllm == base.vllm
        assert omni.patch.install is install_mamba_patches
        assert omni.language_tower.hf_overrides is omni_language_tower_hf_override
        assert omni.language_tower.model_class_overrides == {
            "NemotronHOmniLanguageForCausalLM": (
                "agilerl.architectures.nemotron_h.omni_language:"
                "NemotronHOmniLanguageForCausalLM"
            ),
        }
        assert base.language_tower.hf_overrides is None
        assert base.language_tower.model_class_overrides is None
        assert omni.multimodal_towers_kept_hf_override == {
            "architectures": ["NemotronH_Super_Omni_Reasoning_V3"],
        }
        assert omni.enable_tower_connector_lora is True
        assert base.enable_tower_connector_lora is False

    def test_gemma4_adds_language_tower_without_tower_lora(self) -> None:
        gemma4 = FAMILY_RUNTIME_CONFIGS["gemma4"]
        gemma3 = FAMILY_RUNTIME_CONFIGS["gemma3"]
        assert gemma4.trainer == gemma3.trainer
        assert gemma4.language_tower.hf_overrides is gemma4_language_tower_hf_override
        assert gemma4.language_tower.model_class_overrides is None
        assert gemma4.enable_tower_connector_lora is False
        assert gemma3.language_tower.hf_overrides is None
        assert FAMILY_RUNTIME_CONFIGS["gemma4_text"].language_tower.hf_overrides is (
            gemma4_language_tower_hf_override
        )

    def test_nemotron_h_lookup(self) -> None:
        config = FAMILY_RUNTIME_CONFIGS["nemotron_h"]
        assert config.vllm.model_dump(exclude_none=True) == NEMOTRON_VLLM_KWARGS
        assert config.trainer.model_dump(exclude_none=True) == NEMOTRON_TRAINER_KWARGS
        assert config.patch.install is install_mamba_patches

    def test_only_nemotron_h_omni_trainer_loads_checkpoint_code(self) -> None:
        omni = FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"]

        assert omni.trainer.model_dump(exclude_none=True) == (
            NEMOTRON_OMNI_TRAINER_KWARGS
        )
        assert FAMILY_RUNTIME_CONFIGS["nemotron_h"].trainer.trust_remote_code is None

    def test_nemotron_h_enables_prefix_caching(self) -> None:
        assert FAMILY_RUNTIME_CONFIGS["nemotron_h"].vllm.enable_prefix_caching is True

    @pytest.mark.parametrize("model_type", SWA_MODEL_TYPES)
    def test_swa_lookup(self, model_type: str) -> None:
        assert (
            FAMILY_RUNTIME_CONFIGS[model_type].trainer.model_dump(exclude_none=True)
            == FLEX_TRAINER_KWARGS
        )

    def test_gpt_oss_lookup(self) -> None:
        assert (
            FAMILY_RUNTIME_CONFIGS["gpt_oss"].trainer.model_dump(exclude_none=True)
            == FLEX_TRAINER_KWARGS
        )

    def test_qwen3_5_lookup(self) -> None:
        assert (
            FAMILY_RUNTIME_CONFIGS["qwen3_5"].language_tower.lora_key_prefix
            == "model.language_model.model."
        )

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

    @pytest.mark.parametrize("model_type", ["gemma", "gemma2", "qwen2", "llama"])
    def test_unknown_hub_type_returns_empty_defaults(
        self, monkeypatch: pytest.MonkeyPatch, model_type: str
    ) -> None:
        stub_config_model_type(monkeypatch, model_type)
        config = family_runtime("some/model")
        assert config.vllm.model_dump(exclude_none=True) == {}
        assert config.trainer.model_dump(exclude_none=True) == EMPTY_TRAINER_KWARGS
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
        stub_config_model_type(monkeypatch, "nemotron_h")
        assert pretrained_model_type("nvidia/unlisted-model") == "nemotron_h"

    def test_hub_fallback_sets_trust_remote_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        monkeypatch.setattr(
            "agilerl.architectures.catalog.SUPPORTED_MODEL_INFO",
            {},
        )
        assert (
            pretrained_model_type("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
            == "nemotron_h"
        )
        assert seen["trust_remote_code"] is True

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
            LanguageTowerRuntimeConfig,
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
            (LanguageTowerRuntimeConfig, {}),
            (ModelRuntimeConfig, {}),
        )
        for cls, payload in cases:
            with pytest.raises(ValidationError, match="extra_forbidden"):
                cls.model_validate({**payload, "__unknown__": True})
