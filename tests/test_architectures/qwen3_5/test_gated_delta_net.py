# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Catalog GatedDeltaNet conv-clone wrap."""

from __future__ import annotations

import torch
from torch import nn

from agilerl.architectures.qwen3_5.gated_delta_net import (
    apply_gdn_conv_clone,
    clone_gdn_conv_output,
    install_gdn_conv_clone,
)
from agilerl.architectures.qwen3_5.packed import install_qwen_gdn_patches
from agilerl.architectures.runtime import (
    GdnMixerConfig,
    GdnPatchConfig,
    PatchRuntimeConfig,
)


class DummyGatedDeltaNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def identity(x, **_kwargs):
            return x

        self.causal_conv1d_fn = identity

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class TestCloneGdnConvOutput:
    def test_returns_a_new_tensor_when_the_kernel_is_in_place(self) -> None:
        def identity(x, **_kwargs):
            return x

        wrapped = clone_gdn_conv_output(identity)
        x = torch.zeros(2, 3, requires_grad=True)

        out = wrapped(x=x)

        assert out is not x
        assert torch.equal(out, x)

    def test_returns_the_kernel_output_when_it_needs_no_grad(self) -> None:
        def identity(x, **_kwargs):
            return x

        wrapped = clone_gdn_conv_output(identity)
        x = torch.zeros(2, 3)

        out = wrapped(x=x)

        assert out is x

    def test_is_idempotent(self) -> None:
        def identity(x, **_kwargs):
            return x

        wrapped = clone_gdn_conv_output(identity)

        assert clone_gdn_conv_output(wrapped) is wrapped


class TestInstallGdnConvClone:
    def test_wraps_and_is_idempotent(self) -> None:
        mixer = DummyGatedDeltaNet()
        original = mixer.causal_conv1d_fn
        x = torch.zeros(2, 3, requires_grad=True)

        assert install_gdn_conv_clone(mixer) is True
        assert mixer.causal_conv1d_fn is not original
        assert mixer.causal_conv1d_fn(x=x) is not x
        assert install_gdn_conv_clone(mixer) is False

    def test_skips_when_the_kernel_is_missing(self) -> None:
        mixer = DummyGatedDeltaNet()
        mixer.causal_conv1d_fn = None

        assert install_gdn_conv_clone(mixer) is False


class TestApplyGdnConvClone:
    def test_wraps_catalog_mixer_classes(self) -> None:
        mixer = DummyGatedDeltaNet()
        root = nn.Sequential(mixer)
        path = f"{DummyGatedDeltaNet.__module__}.{DummyGatedDeltaNet.__qualname__}"

        assert apply_gdn_conv_clone(root, (path,)) == 1
        assert mixer._agilerl_gdn_conv_clone is True

    def test_skips_other_modules(self) -> None:
        mixer = DummyGatedDeltaNet()
        root = nn.Sequential(mixer)

        assert apply_gdn_conv_clone(root, ("missing.NoSuchMixer",)) == 0
        assert not getattr(mixer, "_agilerl_gdn_conv_clone", False)


class TestInstallQwenGdnPatches:
    def test_clone_conv_output_uses_catalog_mixer_paths(self) -> None:
        mixer = DummyGatedDeltaNet()
        root = nn.Sequential(mixer)
        path = f"{DummyGatedDeltaNet.__module__}.{DummyGatedDeltaNet.__qualname__}"
        patch = PatchRuntimeConfig(
            gdn=GdnPatchConfig(
                mixers=(GdnMixerConfig(mixer=path, text_model="pkg.TextModel"),)
            )
        )

        install_qwen_gdn_patches(patch, model=root)

        assert mixer._agilerl_gdn_conv_clone is True

    def test_skips_clone_when_the_catalog_flag_is_off(self) -> None:
        mixer = DummyGatedDeltaNet()
        root = nn.Sequential(mixer)
        path = f"{DummyGatedDeltaNet.__module__}.{DummyGatedDeltaNet.__qualname__}"
        patch = PatchRuntimeConfig(
            gdn=GdnPatchConfig(
                mixers=(GdnMixerConfig(mixer=path, text_model="pkg.TextModel"),),
                clone_conv_output=False,
            )
        )

        install_qwen_gdn_patches(patch, model=root)

        assert not getattr(mixer, "_agilerl_gdn_conv_clone", False)

    def test_packed_sequences_use_catalog_mixer_and_text_model(
        self, monkeypatch
    ) -> None:
        seen: list[tuple[str, str]] = []
        monkeypatch.setattr(
            "agilerl.architectures.qwen3_5.packed.patch_qwen_gdn_packed_sequences",
            lambda mixer, text_model: seen.append((mixer, text_model)),
        )
        patch = PatchRuntimeConfig(
            gdn=GdnPatchConfig(
                mixers=(GdnMixerConfig(mixer="pkg.Mixer", text_model="pkg.TextModel"),)
            )
        )

        install_qwen_gdn_patches(patch, use_sequence_packing=True)

        assert seen == [("pkg.Mixer", "pkg.TextModel")]
