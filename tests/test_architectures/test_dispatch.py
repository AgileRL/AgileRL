# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for family runtime lookup and family patch dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agilerl import architectures
from agilerl.architectures.catalog import FAMILY_RUNTIME_CONFIGS
from agilerl.architectures.nemotron_h.mamba import install_mamba_patches
from agilerl.architectures.runtime import (
    MambaPatchConfig,
    ModelRuntimeConfig,
    PatchRuntimeConfig,
)


def recording_install(sink: list[object]):
    def install(patch: PatchRuntimeConfig, *, model=None):
        sink.append((patch, model))

    return install


def loaded_model(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(model_type=model_type))


def stub_auto_config(monkeypatch: pytest.MonkeyPatch, model_type: str) -> None:
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type=model_type),
    )


NEMOTRON_PATCH = FAMILY_RUNTIME_CONFIGS["nemotron_h"].patch


class TestInstallFamilyPatches:
    def test_loaded_nemotron_runs_its_patch(self, monkeypatch) -> None:
        seen: list[object] = []
        actor = loaded_model("nemotron_h")
        monkeypatch.setattr(NEMOTRON_PATCH, "install", recording_install(seen))

        architectures.install_family_patches(actor.config.model_type, actor)

        assert seen == [(NEMOTRON_PATCH, actor)]

    def test_type_lookup_runs_patch_when_no_model(self, monkeypatch) -> None:
        seen: list[object] = []
        monkeypatch.setattr(NEMOTRON_PATCH, "install", recording_install(seen))

        architectures.install_family_patches("nemotron_h")

        assert seen == [(NEMOTRON_PATCH, None)]

    def test_unpatched_type_is_a_no_op(self, monkeypatch) -> None:
        seen: list[object] = []
        actor = loaded_model("llama")
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
            ModelRuntimeConfig(patch=other_patch),
        )

        architectures.install_family_patches(actor.config.model_type, actor)

        assert seen == [(other_patch, actor)]

    def test_type_without_install_leaves_classes_alone(self, monkeypatch) -> None:
        seen: list[object] = []
        monkeypatch.setattr(NEMOTRON_PATCH, "install", recording_install(seen))

        architectures.install_family_patches("llama")

        assert seen == []

    def test_family_without_registered_patches_is_a_no_op(self, monkeypatch) -> None:
        seen: list[object] = []
        monkeypatch.setattr(NEMOTRON_PATCH, "install", recording_install(seen))

        architectures.install_family_patches("unregistered")

        assert seen == []

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
            ModelRuntimeConfig(patch=patch),
        )

        architectures.install_family_patches(actor.config.model_type, actor)

        assert seen == [(patch, actor)]

    def test_nemotron_catalog_install_is_mamba_patches(self) -> None:
        patch = FAMILY_RUNTIME_CONFIGS["nemotron_h"].patch
        assert patch.install is install_mamba_patches
        assert patch.mamba is not None


class TestFamilyRuntime:
    def test_gemma4_trainer_dumps_flex_attention(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_auto_config(monkeypatch, "gemma4")
        config = architectures.family_runtime("google/gemma-4")
        assert config.trainer.model_dump(exclude_none=True) == {
            "attn_implementation": "flex_attention"
        }

    def test_llama_trainer_dumps_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_auto_config(monkeypatch, "llama")
        assert (
            architectures.family_runtime("meta/llama").trainer.model_dump(
                exclude_none=True
            )
            == {}
        )

    def test_nemotron_vllm_dumps_engine_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_auto_config(monkeypatch, "nemotron_h")
        assert architectures.family_runtime("nvidia/nemotron").vllm.model_dump(
            exclude_none=True
        ) == {
            "mamba_cache_mode": "align",
            "max_num_batched_tokens": 8192,
            "reasoning_parser": "nemotron_v3",
            "enable_prefix_caching": True,
        }

    def test_llama_vllm_dumps_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_auto_config(monkeypatch, "llama")
        assert (
            architectures.family_runtime("meta/llama").vllm.model_dump(
                exclude_none=True
            )
            == {}
        )
