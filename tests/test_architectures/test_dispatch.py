# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for family runtime lookup and family patch dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from agilerl import architectures
from agilerl.architectures.catalog import FAMILY_RUNTIME_CONFIGS, model_type_vision
from agilerl.architectures.gemma4.vision import install_gemma4_vision
from agilerl.architectures.nemotron_h.patches import install_nemotron_h_patches
from agilerl.architectures.qwen3_5.patches import install_qwen3_5_patches
from agilerl.architectures.qwen3_5.vision import install_qwen3_5_vision
from agilerl.architectures.runtime import (
    MambaPatchConfig,
    ModelRuntimeConfig,
    ModelTypes,
    PatchRuntimeConfig,
    VisionSupport,
)

INSTALL_ORDER: list[str] = []


def record_liger() -> None:
    INSTALL_ORDER.append("liger")


def recording_install(sink: list[object]):
    def install(patch: PatchRuntimeConfig, model=None, use_sequence_packing=False):
        sink.append((patch, model, use_sequence_packing))

    return install


def loaded_model(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(model_type=model_type))


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


NEMOTRON_PATCH = FAMILY_RUNTIME_CONFIGS["nemotron_h"].patch


class TestInstallFamilyPatches:
    def test_loaded_nemotron_runs_its_patch(self, monkeypatch) -> None:
        seen: list[object] = []
        actor = loaded_model("nemotron_h")
        monkeypatch.setattr(NEMOTRON_PATCH, "install", recording_install(seen))

        architectures.install_family_patches(actor.config.model_type, actor)

        assert seen == [(NEMOTRON_PATCH, actor, False)]

    def test_type_lookup_runs_patch_when_no_model(self, monkeypatch) -> None:
        seen: list[object] = []
        monkeypatch.setattr(NEMOTRON_PATCH, "install", recording_install(seen))

        architectures.install_family_patches("nemotron_h")

        assert seen == [(NEMOTRON_PATCH, None, False)]

    def test_unpatched_type_is_a_no_op(self, monkeypatch) -> None:
        seen: list[object] = []
        actor = loaded_model("qwen3")
        monkeypatch.setattr(NEMOTRON_PATCH, "install", recording_install(seen))

        architectures.install_family_patches(actor.config.model_type, actor)

        assert seen == []

    def test_custom_install_is_invoked(self, monkeypatch) -> None:
        seen: list[object] = []
        actor = loaded_model("custom")
        other_patch = PatchRuntimeConfig(install=recording_install(seen))
        monkeypatch.setitem(
            architectures.FAMILY_RUNTIME_CONFIGS,
            "custom",
            ModelRuntimeConfig(
                name="custom",
                model_types=ModelTypes(text=("custom",)),
                patch=other_patch,
            ),
        )

        architectures.install_family_patches(actor.config.model_type, actor)

        assert seen == [(other_patch, actor, False)]

    def test_type_without_install_leaves_classes_alone(self, monkeypatch) -> None:
        seen: list[object] = []
        monkeypatch.setattr(NEMOTRON_PATCH, "install", recording_install(seen))

        architectures.install_family_patches("qwen3")

        assert seen == []

    @pytest.mark.parametrize("model_type", [None, ""])
    def test_missing_model_type_is_a_no_op(self, monkeypatch, model_type) -> None:
        seen: list[object] = []
        monkeypatch.setattr(NEMOTRON_PATCH, "install", recording_install(seen))

        architectures.install_family_patches(model_type)

        assert seen == []

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="'phi3' has no family"):
            architectures.install_family_patches("phi3")

    def test_loaded_gemma_does_not_patch(self, monkeypatch) -> None:
        seen: list[object] = []
        actor = loaded_model("gemma4")
        monkeypatch.setattr(NEMOTRON_PATCH, "install", recording_install(seen))

        architectures.install_family_patches(actor.config.model_type, actor)

        assert seen == []

    def test_catalog_patch_is_passed_to_installer(self, monkeypatch) -> None:
        seen: list[object] = []
        actor = loaded_model("nemotron_h")
        patch = PatchRuntimeConfig(
            install=recording_install(seen),
            mamba=MambaPatchConfig(
                mixer="pkg.Mixer", fused_path=False, stream_ordering=True
            ),
        )
        monkeypatch.setitem(
            architectures.FAMILY_RUNTIME_CONFIGS,
            "nemotron_h",
            ModelRuntimeConfig(
                name="nemotron_h",
                model_types=ModelTypes(text=("nemotron_h",)),
                patch=patch,
            ),
        )

        architectures.install_family_patches(actor.config.model_type, actor)

        assert seen == [(patch, actor, False)]

    def test_runs_liger_then_text_then_vision_patches(self, monkeypatch) -> None:
        # Arrange
        INSTALL_ORDER.clear()

        def text_install(_patch, _model=None, _use_sequence_packing=False) -> None:
            INSTALL_ORDER.append("text")

        def vision_install(_patch, _model=None, _use_sequence_packing=False) -> None:
            INSTALL_ORDER.append("vision")

        monkeypatch.setattr(architectures, "HAS_LIGER_KERNEL", True)
        monkeypatch.setitem(
            architectures.FAMILY_RUNTIME_CONFIGS,
            "custom_vl",
            ModelRuntimeConfig(
                name="custom",
                model_types=ModelTypes(multimodal=("custom_vl",)),
                patch=PatchRuntimeConfig(install=text_install),
                liger="tests.test_architectures.test_dispatch:record_liger",
                vision=VisionSupport(install=vision_install),
            ),
        )

        # Act
        architectures.install_family_patches("custom_vl")

        # Assert
        assert INSTALL_ORDER == ["liger", "text", "vision"]

    def test_text_model_type_skips_the_vision_patch(self, monkeypatch) -> None:
        # Arrange
        seen: list[object] = []
        family = ModelRuntimeConfig(
            name="custom",
            model_types=ModelTypes(text=("custom_text",), multimodal=("custom_vl",)),
            vision=VisionSupport(install=recording_install(seen)),
        )
        monkeypatch.setitem(architectures.FAMILY_RUNTIME_CONFIGS, "custom_text", family)
        monkeypatch.setitem(architectures.FAMILY_RUNTIME_CONFIGS, "custom_vl", family)

        # Act
        architectures.install_family_patches("custom_text")
        text_calls = list(seen)
        architectures.install_family_patches("custom_vl")

        # Assert
        assert text_calls == []
        assert seen == [(family.patch, None, False)]

    def test_liger_is_skipped_without_liger_kernel(self, monkeypatch) -> None:
        INSTALL_ORDER.clear()
        monkeypatch.setattr(architectures, "HAS_LIGER_KERNEL", False)
        monkeypatch.setitem(
            architectures.FAMILY_RUNTIME_CONFIGS,
            "custom",
            ModelRuntimeConfig(
                name="custom",
                model_types=ModelTypes(text=("custom",)),
                liger="tests.test_architectures.test_dispatch:record_liger",
            ),
        )

        architectures.install_family_patches("custom")

        assert INSTALL_ORDER == []

    def test_nemotron_catalog_install_is_mamba_patches(self) -> None:
        patch = FAMILY_RUNTIME_CONFIGS["nemotron_h"].patch
        assert patch.install is install_nemotron_h_patches
        assert patch.mamba is not None

    def test_qwen_vl_catalog_install_adds_vision_rows(self) -> None:
        for model_type in ("qwen3_5", "qwen3_5_moe"):
            vision = model_type_vision(model_type)
            assert vision is not None, model_type
            assert vision.install is install_qwen3_5_vision, model_type

    def test_gemma4_vl_catalog_install_adds_vision_rows(self) -> None:
        vision = model_type_vision("gemma4")
        assert vision is not None
        assert vision.install is install_gemma4_vision
        assert FAMILY_RUNTIME_CONFIGS["gemma4"].patch.install is None
        assert model_type_vision("gemma4_text") is None

    def test_qwen_catalog_install_is_gdn_packed_patches(self) -> None:
        for model_type in (
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_5_text",
            "qwen3_5_moe_text",
        ):
            patch = FAMILY_RUNTIME_CONFIGS[model_type].patch
            assert patch.install is install_qwen3_5_patches, model_type

    def test_qwen_catalog_lists_gdn_mixers_and_checkpoint_children(self) -> None:
        for model_type in (
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_5_text",
            "qwen3_5_moe_text",
        ):
            patch = FAMILY_RUNTIME_CONFIGS[model_type].patch
            assert patch.gdn is not None, model_type
            assert patch.gdn.clone_conv_output is True
            assert tuple(
                (entry.mixer, entry.text_model) for entry in patch.gdn.mixers
            ) == (
                (
                    (
                        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
                        "Qwen3_5MoeGatedDeltaNet"
                    ),
                    (
                        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
                        "Qwen3_5MoeTextModel"
                    ),
                ),
                (
                    "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet",
                    "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel",
                ),
            )


class TestFamilyRuntime:
    def test_gemma4_trainer_builds_the_vl_class(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_config_model_type(monkeypatch, "gemma4")
        config = architectures.family_runtime("google/gemma-4")
        assert architectures.trainer_model_class("google/gemma-4") is (
            AutoModelForImageTextToText
        )
        assert config.trainer.load_kwargs() == {}

    def test_gpt_oss_trainer_builds_a_causal_lm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_config_model_type(monkeypatch, "gpt_oss")
        config = architectures.family_runtime("openai/gpt-oss-20b")
        assert architectures.trainer_model_class("openai/gpt-oss-20b") is (
            AutoModelForCausalLM
        )
        assert config.trainer.load_kwargs() == {}

    def test_nemotron_trainer_dumps_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_config_model_type(monkeypatch, "nemotron_h")
        config = architectures.family_runtime("nvidia/nemotron")
        assert config.trainer.load_kwargs() == {}

    def test_qwen3_trainer_dumps_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_config_model_type(monkeypatch, "qwen3")
        trainer = architectures.family_runtime("Qwen/Qwen3-4B").trainer
        assert (
            architectures.trainer_model_class("Qwen/Qwen3-4B") is AutoModelForCausalLM
        )
        assert trainer.load_kwargs() == {}

    def test_nemotron_vllm_dumps_engine_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_config_model_type(monkeypatch, "nemotron_h")
        assert architectures.family_runtime("nvidia/nemotron").vllm.engine_kwargs() == {
            "mamba_cache_mode": "align",
            "max_num_batched_tokens": 8192,
            "enable_prefix_caching": True,
            "prefix_cache_retention_interval": None,
        }

    def test_qwen3_vllm_dumps_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_config_model_type(monkeypatch, "qwen3")
        assert architectures.family_runtime("Qwen/Qwen3-4B").vllm.engine_kwargs() == {}
