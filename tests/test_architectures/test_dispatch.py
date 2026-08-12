# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for family detection and the ZeRO-3 family patch dispatch."""

import pytest

from agilerl import architectures


class TestDetectModelFamilies:
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
        assert architectures.detect_model_families(name) == frozenset({"nemotron_h"})

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
        assert architectures.detect_model_families(name) == frozenset()


class TestInstallFamilyZero3Patches:
    def test_detected_family_runs_every_patch_it_declares(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setitem(
            architectures.ZERO3_FAMILY_PATCHES,
            "nemotron_h",
            (lambda: calls.append("fused"), lambda: calls.append("stream")),
        )

        patched = architectures.install_family_zero3_patches("nvidia/Nemotron-H-8B")

        assert patched == frozenset({"nemotron_h"})
        assert calls == ["fused", "stream"]

    def test_undetected_family_leaves_its_classes_alone(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setitem(
            architectures.ZERO3_FAMILY_PATCHES,
            "nemotron_h",
            (lambda: calls.append("fused"),),
        )

        patched = architectures.install_family_zero3_patches("meta-llama/Llama-3.1-8B")

        assert patched == frozenset()
        assert calls == []

    def test_family_without_registered_patches_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(
            architectures,
            "detect_model_families",
            lambda _name: frozenset({"unregistered"}),
        )

        assert architectures.install_family_zero3_patches("whatever") == frozenset(
            {"unregistered"},
        )

    def test_real_nemotron_h_entry_declares_both_mixer_patches(self):
        from agilerl.architectures import nemotron_h

        assert set(architectures.ZERO3_FAMILY_PATCHES["nemotron_h"]) == {
            nemotron_h.patch_nemotron_mamba_fused_path,
            nemotron_h.patch_nemotron_mamba_stream_ordering,
        }
