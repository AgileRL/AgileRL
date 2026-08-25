# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Architecture-scoped model patches, dispatched by family.

A family is a stable key (``nemotron_h``) detected from the checkpoint id.
:func:`install_family_zero3_patches` runs the patches a family needs before
its model is built; :data:`ZERO3_FAMILY_PATCHES` is the table it reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agilerl.architectures.nemotron_h import (
    patch_nemotron_mamba_fused_path,
    patch_nemotron_mamba_stream_ordering,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from agilerl.protocols import PreTrainedModelProtocol

__all__ = [
    "ZERO3_FAMILY_PATCHES",
    "detect_model_families",
    "install_family_zero3_patches",
]

ZERO3_FAMILY_PATCHES: Mapping[str, tuple[Callable[..., object], ...]] = {
    "nemotron_h": (
        patch_nemotron_mamba_fused_path,
        patch_nemotron_mamba_stream_ordering,
    ),
}


def detect_model_families(model_name_or_path: str | None) -> frozenset[str]:
    """Detect architecture families from a Hugging Face id or local path.

    Hybrid Nano / Nemotron-H checkpoints often omit ``H`` from the repo name
    (e.g. ``NVIDIA-Nemotron-3-Nano-30B-A3B-BF16``) even when
    ``model_type == "nemotron_h"``, so the substring is intentionally broad.

    :param model_name_or_path: Hugging Face id or local path, or None.
    :type model_name_or_path: str | None
    :return: Family keys that should receive patches.
    :rtype: frozenset[str]
    """
    if not model_name_or_path:
        return frozenset()
    normalized = model_name_or_path.replace("\\", "/").lower()
    families: set[str] = set()
    if "nemotron" in normalized:
        families.add("nemotron_h")
    return frozenset(families)


def install_family_zero3_patches(
    model_name_or_path: str | None,
    *,
    model: PreTrainedModelProtocol | None = None,
) -> frozenset[str]:
    """Install the ZeRO-3 patches every detected family needs.

    Architectures other than the detected ones never have their classes
    mutated, so an unrelated model cannot fail on a shape skew. Call before
    the model is built; pass ``model`` to also fix instances that already
    exist.

    :param model_name_or_path: Hugging Face id or local path, or None.
    :type model_name_or_path: str | None
    :param model: Already-built model the patches also apply to, or None.
    :type model: PreTrainedModelProtocol | None
    :return: The families that were patched.
    :rtype: frozenset[str]
    """
    families = detect_model_families(model_name_or_path)
    for family in sorted(families):
        for patch in ZERO3_FAMILY_PATCHES.get(family, ()):
            patch(model=model)
    return families
