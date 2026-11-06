# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Granite 3.1 MoE: Liger registration and tensor-parallel attention and tied embedding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agilerl import HAS_LIGER_KERNEL
from agilerl.architectures.runtime import PatchRuntimeConfig

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel

__all__ = ["install_granite_moe_patches"]


def install_granite_moe_patches(
    _patch: PatchRuntimeConfig,
    _model: PreTrainedModel | PeftModel | None = None,
    _use_sequence_packing: bool = False,
) -> None:
    """Register the GraniteMoe Liger applier when liger-kernel is installed.

    :param _patch: Catalog patch config.
    :type _patch: PatchRuntimeConfig
    :param _model: Already-built model, or None; Liger patches it later.
    :type _model: PreTrainedModel | PeftModel | None
    :param _use_sequence_packing: Unused; Granite MoE has no packed-sequence wrap.
    :type _use_sequence_packing: bool, optional
    :return: None
    :rtype: None
    """
    if HAS_LIGER_KERNEL:
        # Defers liger-kernel, an optional extra.
        from agilerl.architectures.granite_liger import register_granite_liger

        register_granite_liger()
