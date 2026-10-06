# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.architectures.nemotron_h.mamba import (
    FUSED_PATH_PATCHED_FLAG,
    STREAM_PATCHED_FLAG,
)
from agilerl.utils.llm_packing import RESETS_AT_DOCUMENT_BOUNDARY

if HAS_LLM_DEPENDENCIES:
    from transformers.models.nemotron_h import modeling_nemotron_h
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        NemotronHBlock,
        NemotronHMamba2Mixer,
    )

KERNEL_GLOBALS = ("mamba_split_conv1d_scan_combined", "mamba_chunk_scan_combined")
MIXER_METHODS = ("__init__", "forward", "torch_forward", "cuda_kernels_forward")
MIXER_FLAGS = (
    FUSED_PATH_PATCHED_FLAG,
    STREAM_PATCHED_FLAG,
    RESETS_AT_DOCUMENT_BOUNDARY,
)


@pytest.fixture
def pristine_nemotron_classes():
    """Restore the Nemotron-H mixer and block classes and the kernel globals.

    Kernel globals are absent until a mixer builds.
    """
    saved_mixer = {name: NemotronHMamba2Mixer.__dict__[name] for name in MIXER_METHODS}
    saved_block_forward = NemotronHBlock.__dict__["forward"]
    saved_kernels = {
        name: getattr(modeling_nemotron_h, name)
        for name in KERNEL_GLOBALS
        if hasattr(modeling_nemotron_h, name)
    }
    yield
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
