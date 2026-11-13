# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H text-training patches: the catalog Mamba2 mixer workarounds."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agilerl.architectures.nemotron_h.layers import (
    patch_nemotron_mamba_fused_path,
    patch_nemotron_mamba_packed_sequences,
    patch_nemotron_mamba_stream_ordering,
    patch_sdpa_fully_masked_rows,
)
from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.utils.patching import library_shim

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel

__all__ = ["install_nemotron_h_patches"]


@library_shim("transformers", "==5.19.*")
def install_nemotron_h_patches(
    patch: PatchRuntimeConfig,
    model: PreTrainedModel | PeftModel | None = None,
    use_sequence_packing: bool = False,
) -> None:
    """Install catalog Mamba2 mixer workarounds.

    When ``use_sequence_packing`` is set, ``block`` routes packed-row
    ``seq_idx`` into the mixer so its state resets at each document boundary.
    ``fused_path`` gathers SSM kernel parameters.
    ``stream_ordering``
    wraps mixer ``forward`` so default-stream scan/conv kernels wait on the
    caller stream. Those kernels launch on the default stream, so a caller on
    another stream can race with a parameter all-gather. When ``model`` is
    set, attention modules whose forward calls scaled-dot-product attention
    replace NaN rows from a fully masked mask.

    :param patch: Catalog patch config; mamba flags and mixer class path.
    :type patch: PatchRuntimeConfig
    :param model: Already-built model the patches also apply to, or None.
    :type model: PreTrainedModel | PeftModel | None
    :param use_sequence_packing: Install the packed-sequence wrap.
    :type use_sequence_packing: bool, optional
    :return: None
    :rtype: None
    """
    if patch.mamba is None:
        return
    # Installed first so the fused-path and stream wrappers wrap the packed forward.
    if use_sequence_packing and patch.mamba.block is not None:
        patch_nemotron_mamba_packed_sequences(
            mixer=patch.mamba.mixer, block=patch.mamba.block
        )
    if patch.mamba.fused_path:
        patch_nemotron_mamba_fused_path(mixer=patch.mamba.mixer, model=model)
    if patch.mamba.stream_ordering:
        patch_nemotron_mamba_stream_ordering(mixer=patch.mamba.mixer, model=model)
    if model is not None:
        patch_sdpa_fully_masked_rows(model)
