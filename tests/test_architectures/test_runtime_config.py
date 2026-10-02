# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for model-type trainer and vLLM runtime configs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agilerl.architectures.catalog import (
    FAMILY_RUNTIME_CONFIGS,
    family_runtime,
    pretrained_model_type,
)
from agilerl.architectures.nemotron_h.language_tower import (
    omni_language_tower_hf_override,
)
from agilerl.architectures.nemotron_h.mamba import install_mamba_patches

NEMOTRON_VLLM_KWARGS = {
    "mamba_cache_mode": "align",
    "max_num_batched_tokens": 8192,
    "reasoning_parser": "nemotron_v3",
    "enable_prefix_caching": True,
    "trust_remote_code": True,
}

NEMOTRON_TRAINER_KWARGS = {
    "attn_implementation": "flash_attention_2",
    "trust_remote_code": True,
}
GEMMA_TRAINER_KWARGS = {"attn_implementation": "flex_attention"}
EMPTY_TRAINER_KWARGS: dict[str, object] = {}
FLEX_TRAINER_KWARGS = GEMMA_TRAINER_KWARGS

SWA_MODEL_TYPES = ("gemma3", "gemma3_text", "gemma4", "gemma4_text")


def stub_config_model_type(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    auto_map: dict[str, str] | None = None,
) -> None:
    config: dict[str, object] = {"model_type": model_type}
    if auto_map is not None:
        config["auto_map"] = auto_map

    @classmethod
    def fake_get_config_dict(
        cls, pretrained_model_name_or_path: str, **kwargs: object
    ) -> tuple[dict[str, object], dict[str, object]]:
        return (config, {})

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
        }

    def test_nemotron_h_omni_keeps_nemotron_h_runtime_and_adds_language_tower(
        self,
    ) -> None:
        base = FAMILY_RUNTIME_CONFIGS["nemotron_h"]
        omni = FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"]
        assert omni.trainer == base.trainer
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

    def test_nemotron_h_lookup(self) -> None:
        config = FAMILY_RUNTIME_CONFIGS["nemotron_h"]
        assert config.vllm.model_dump(exclude_none=True) == NEMOTRON_VLLM_KWARGS
        assert config.trainer.model_dump(exclude_none=True) == NEMOTRON_TRAINER_KWARGS
        assert config.patch.install is install_mamba_patches

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

    @pytest.mark.parametrize("model_type", ["nemotron_h", "nemotron_h_omni"])
    def test_remote_modeling_downgrades_flash_attention_2_to_eager(
        self, monkeypatch: pytest.MonkeyPatch, model_type: str
    ) -> None:
        stub_config_model_type(
            monkeypatch,
            model_type,
            auto_map={
                "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM"
            },
        )

        config = family_runtime("nvidia/nemotron")

        assert config.trainer.attn_implementation == "eager"
        assert config.trainer.trust_remote_code is True
        assert (
            FAMILY_RUNTIME_CONFIGS[model_type].trainer.attn_implementation
            == "flash_attention_2"
        )

    def test_remote_modeling_downgrades_flex_attention_to_eager(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_config_model_type(
            monkeypatch,
            "gemma4",
            auto_map={"AutoModelForCausalLM": "modeling_gemma4.Gemma4ForCausalLM"},
        )

        config = family_runtime("google/gemma")

        assert config.trainer.attn_implementation == "eager"

    def test_remote_modeling_without_family_attn_stays_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_config_model_type(
            monkeypatch,
            "llama",
            auto_map={"AutoModelForCausalLM": "modeling_llama.LlamaForCausalLM"},
        )

        config = family_runtime("meta/llama")

        assert config.trainer.model_dump(exclude_none=True) == EMPTY_TRAINER_KWARGS

    def test_supported_remote_id_skips_hub_lookup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_if_called(*args: object, **kwargs: object) -> None:
            msg = "hub lookup"
            raise AssertionError(msg)

        monkeypatch.setattr(
            "agilerl.architectures.catalog.PretrainedConfig.get_config_dict",
            raise_if_called,
        )

        config = family_runtime("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")

        assert config.trainer.attn_implementation == "eager"
        assert config.trainer.trust_remote_code is True

    def test_supported_stock_id_skips_hub_lookup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_if_called(*args: object, **kwargs: object) -> None:
            msg = "hub lookup"
            raise AssertionError(msg)

        monkeypatch.setattr(
            "agilerl.architectures.catalog.PretrainedConfig.get_config_dict",
            raise_if_called,
        )

        config = family_runtime("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")

        assert config.trainer.attn_implementation == "flash_attention_2"


class TestPretrainedModelType:
    def test_reads_config_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_config_model_type(monkeypatch, "nemotron_h")
        assert pretrained_model_type("nvidia/unlisted-model") == "nemotron_h"

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
