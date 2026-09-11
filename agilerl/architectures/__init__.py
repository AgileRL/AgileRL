# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Architecture-scoped model patches, dispatched by family.

A family is a stable key (``nemotron_h``) detected from the checkpoint id.
:func:`install_family_patches` runs the family's installer with the run's
ZeRO stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agilerl.architectures.nemotron_h import install_nemotron_h_patches

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from agilerl.protocols import PreTrainedModelProtocol

__all__ = [
    "FAMILY_PATCHES",
    "detect_model_family",
    "install_family_patches",
]

FAMILY_PATCHES: Mapping[str, Callable[..., object]] = {
    "nemotron_h": install_nemotron_h_patches,
}


def detect_model_family(model_name_or_path: str | None) -> str | None:
    """Detect the architecture family from a Hugging Face id or local path.

    Hybrid Nano / Nemotron-H checkpoints often omit ``H`` from the repo name
    (e.g. ``NVIDIA-Nemotron-3-Nano-30B-A3B-BF16``) even when
    ``model_type == "nemotron_h"``, so the substring is intentionally broad.

    :param model_name_or_path: Hugging Face id or local path, or None.
    :type model_name_or_path: str | None
    :return: Family key that should receive patches, or None.
    :rtype: str | None
    """
    if not model_name_or_path:
        return None
    normalized = model_name_or_path.replace("\\", "/").lower()
    if "nemotron" in normalized:
        return "nemotron_h"
    return None


def install_family_patches(
    model_name_or_path: str | None,
    *,
    zero_stage: int,
    model: PreTrainedModelProtocol | None = None,
) -> str | None:
    """Install the detected family's patches for this ZeRO stage.

    :param model_name_or_path: Hugging Face id or local path, or None.
    :type model_name_or_path: str | None
    :param zero_stage: DeepSpeed ZeRO stage for this run.
    :type zero_stage: int
    :param model: Already-built model the patches also apply to, or None.
    :type model: PreTrainedModelProtocol | None
    :return: The family that was patched, or None.
    :rtype: str | None
    """
    family = detect_model_family(model_name_or_path)
    if family is None:
        return None
    patch = FAMILY_PATCHES.get(family)
    if patch is not None:
        patch(model=model, zero_stage=zero_stage)
    return family
