# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Reset the class-level Qwen GDN packed-sequence patches to the unpatched HF state."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextModel,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeTextModel,
)

from agilerl.architectures.qwen3_5.packed import VARLEN_SCAN
from agilerl.utils.llm_packing import RESETS_AT_DOCUMENT_BOUNDARY

# Captured at import, before any test installs the family patches.
PRISTINE_GDN_FORWARDS = tuple(
    (cls, cls.__dict__["forward"])
    for cls in (
        Qwen3_5MoeGatedDeltaNet,
        Qwen3_5MoeTextModel,
        Qwen3_5GatedDeltaNet,
        Qwen3_5TextModel,
    )
)
GDN_MIXERS = (Qwen3_5MoeGatedDeltaNet, Qwen3_5GatedDeltaNet)


def reset_qwen_gdn_classes() -> None:
    """Put the unpatched GDN mixer and text model forwards back."""
    for cls, forward in PRISTINE_GDN_FORWARDS:
        cls.forward = forward
    for mixer in GDN_MIXERS:
        if RESETS_AT_DOCUMENT_BOUNDARY in vars(mixer):
            delattr(mixer, RESETS_AT_DOCUMENT_BOUNDARY)


@contextmanager
def restored_qwen_gdn_classes() -> Iterator[None]:
    """Run with the unpatched Qwen GDN forwards; on exit reset them and restore the GDN scan."""
    # The scan is restored, not reset: the torch_kernels fixture may hold it at its PyTorch fallback.
    scans = [
        (
            sys.modules[mixer.__module__],
            getattr(sys.modules[mixer.__module__], VARLEN_SCAN),
        )
        for mixer in GDN_MIXERS
    ]
    reset_qwen_gdn_classes()
    try:
        yield
    finally:
        reset_qwen_gdn_classes()
        for module, scan in scans:
            setattr(module, VARLEN_SCAN, scan)
