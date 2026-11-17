# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Restore the transformers Nemotron-H classes that the Mamba patches change."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from transformers.models.nemotron_h import modeling_nemotron_h
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHBlock,
    NemotronHMamba2Mixer,
)

from agilerl.architectures.nemotron_h.mamba import (
    FUSED_PATH_PATCHED_FLAG,
    STREAM_PATCHED_FLAG,
)
from agilerl.utils.llm_packing import RESETS_AT_DOCUMENT_BOUNDARY

KERNEL_GLOBALS = ("mamba2_split_conv1d_scan_combined", "mamba2_chunk_scan")
MIXER_METHODS = ("__init__", "forward")
MIXER_FLAGS = (
    FUSED_PATH_PATCHED_FLAG,
    STREAM_PATCHED_FLAG,
    RESETS_AT_DOCUMENT_BOUNDARY,
)


@contextmanager
def restore_nemotron_classes() -> Iterator[None]:
    """Restore the Nemotron-H mixer and block classes and the kernel globals on exit.

    Kernel globals are absent until a mixer builds.
    """
    saved_mixer = {name: NemotronHMamba2Mixer.__dict__[name] for name in MIXER_METHODS}
    saved_block_forward = NemotronHBlock.__dict__["forward"]
    saved_kernels = {
        name: getattr(modeling_nemotron_h, name)
        for name in KERNEL_GLOBALS
        if hasattr(modeling_nemotron_h, name)
    }
    try:
        yield
    finally:
        for name, method in saved_mixer.items():
            setattr(NemotronHMamba2Mixer, name, method)
        NemotronHBlock.forward = saved_block_forward
        for name in KERNEL_GLOBALS:
            if name in saved_kernels:
                setattr(modeling_nemotron_h, name, saved_kernels[name])
            elif hasattr(modeling_nemotron_h, name):
                delattr(modeling_nemotron_h, name)
        for flag in MIXER_FLAGS:
            if flag in vars(NemotronHMamba2Mixer):
                delattr(NemotronHMamba2Mixer, flag)
