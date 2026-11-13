# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from agilerl.architectures.nemotron_h import patches
from agilerl.architectures.runtime import MambaPatchConfig, PatchRuntimeConfig

MIXER = "transformers.models.nemotron_h.modeling_nemotron_h.NemotronHMamba2Mixer"


class TestInstallNemotronHPatches:
    def test_runs_fused_path_and_stream_ordering(self, monkeypatch):
        seen: list[tuple[str, object]] = []
        actor = torch.nn.Module()
        monkeypatch.setattr(
            patches,
            "patch_nemotron_mamba_fused_path",
            lambda *, mixer, model=None: seen.append(("fused", mixer, model)),
        )
        monkeypatch.setattr(
            patches,
            "patch_nemotron_mamba_stream_ordering",
            lambda *, mixer, model=None: seen.append(("stream", mixer, model)),
        )

        patches.install_nemotron_h_patches(
            PatchRuntimeConfig(mamba=MambaPatchConfig(mixer=MIXER)), model=actor
        )

        assert seen == [("fused", MIXER, actor), ("stream", MIXER, actor)]

    def test_fused_path_false_skips_fused_path(self, monkeypatch):
        seen: list[tuple[str, object]] = []
        actor = torch.nn.Module()
        monkeypatch.setattr(
            patches,
            "patch_nemotron_mamba_fused_path",
            lambda *, mixer, model=None: seen.append(("fused", mixer, model)),
        )
        monkeypatch.setattr(
            patches,
            "patch_nemotron_mamba_stream_ordering",
            lambda *, mixer, model=None: seen.append(("stream", mixer, model)),
        )

        patches.install_nemotron_h_patches(
            PatchRuntimeConfig(mamba=MambaPatchConfig(mixer=MIXER, fused_path=False)),
            model=actor,
        )

        assert seen == [("stream", MIXER, actor)]

    def test_stream_ordering_false_skips_stream_ordering(self, monkeypatch):
        seen: list[tuple[str, object]] = []
        actor = torch.nn.Module()
        monkeypatch.setattr(
            patches,
            "patch_nemotron_mamba_fused_path",
            lambda *, mixer, model=None: seen.append(("fused", mixer, model)),
        )
        monkeypatch.setattr(
            patches,
            "patch_nemotron_mamba_stream_ordering",
            lambda *, mixer, model=None: seen.append(("stream", mixer, model)),
        )

        patches.install_nemotron_h_patches(
            PatchRuntimeConfig(
                mamba=MambaPatchConfig(mixer=MIXER, stream_ordering=False)
            ),
            model=actor,
        )

        assert seen == [("fused", MIXER, actor)]

    def test_missing_mamba_is_a_no_op(self, monkeypatch):
        seen: list[tuple[str, object]] = []
        monkeypatch.setattr(
            patches,
            "patch_nemotron_mamba_fused_path",
            lambda *, mixer, model=None: seen.append(("fused", mixer, model)),
        )
        monkeypatch.setattr(
            patches,
            "patch_nemotron_mamba_stream_ordering",
            lambda *, mixer, model=None: seen.append(("stream", mixer, model)),
        )

        patches.install_nemotron_h_patches(PatchRuntimeConfig(), model=object())

        assert seen == []

    def test_skips_the_packed_wrap_unless_sequence_packing_is_on(self, monkeypatch):
        seen: list[str] = []
        actor = torch.nn.Module()
        block = "transformers.models.nemotron_h.modeling_nemotron_h.NemotronHBlock"
        monkeypatch.setattr(
            patches,
            "patch_nemotron_mamba_packed_sequences",
            lambda *, mixer, block: seen.append("packed"),
        )
        monkeypatch.setattr(
            patches,
            "patch_nemotron_mamba_fused_path",
            lambda *, mixer, model=None: seen.append("fused"),
        )
        monkeypatch.setattr(
            patches,
            "patch_nemotron_mamba_stream_ordering",
            lambda *, mixer, model=None: seen.append("stream"),
        )
        config = PatchRuntimeConfig(mamba=MambaPatchConfig(mixer=MIXER, block=block))

        patches.install_nemotron_h_patches(config, model=actor)

        assert seen == ["fused", "stream"]

        patches.install_nemotron_h_patches(
            config, model=actor, use_sequence_packing=True
        )

        assert seen == ["fused", "stream", "packed", "fused", "stream"]
