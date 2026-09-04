# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for family detection and family patch dispatch."""

import pytest

from agilerl import architectures


class TestDetectModelFamily:
    @pytest.mark.parametrize(
        "name",
        [
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/",
            "nvidia/Nemotron-H-8B-Base-8K",
            "/ckpt/nemotron_local",
            r"C:\ckpt\Nemotron-H-8B",
        ],
    )
    def test_nemotron_ids_resolve_to_nemotron_h(self, name):
        assert architectures.detect_model_family(name) == "nemotron_h"

    @pytest.mark.parametrize(
        "name",
        [
            None,
            "",
            "mock-model",
            "meta-llama/Llama-3.1-8B",
            "google/gemma-4-E4B-it",
        ],
    )
    def test_other_ids_resolve_to_nothing(self, name):
        assert architectures.detect_model_family(name) is None


class TestInstallFamilyPatches:
    def test_detected_family_runs_its_patch(self, monkeypatch):
        seen: list[tuple[object, int]] = []
        actor = object()
        monkeypatch.setitem(
            architectures.FAMILY_PATCHES,
            "nemotron_h",
            lambda *, model=None, zero_stage: seen.append((model, zero_stage)),
        )

        patched = architectures.install_family_patches(
            "nvidia/Nemotron-H-8B",
            zero_stage=2,
            model=actor,
        )

        assert patched == "nemotron_h"
        assert seen == [(actor, 2)]

    def test_undetected_family_leaves_its_classes_alone(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setitem(
            architectures.FAMILY_PATCHES,
            "nemotron_h",
            lambda *, model=None, zero_stage: calls.append("nemo"),
        )

        patched = architectures.install_family_patches(
            "meta-llama/Llama-3.1-8B",
            zero_stage=3,
        )

        assert patched is None
        assert calls == []

    def test_family_without_registered_patches_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(
            architectures,
            "detect_model_family",
            lambda _name: "unregistered",
        )

        assert (
            architectures.install_family_patches("whatever", zero_stage=3)
            == "unregistered"
        )

    def test_nemotron_h_entry_is_install_nemotron_h_patches(self):
        from agilerl.architectures.nemotron_h import install_nemotron_h_patches

        assert architectures.FAMILY_PATCHES["nemotron_h"] is install_nemotron_h_patches
