# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Run tiny CPU Qwen3.5 and Nemotron-H models on transformers' PyTorch kernels.

transformers binds the causal-conv1d, FLA and mamba-ssm kernels when it imports
these modules, if those packages are installed, and the kernels take CUDA tensors only.
"""

from __future__ import annotations

import inspect

import pytest
from transformers.models.nemotron_h import modeling_nemotron_h
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

GDN_KERNELS = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
)

KERNEL_GLOBALS = {
    modeling_qwen3_5: GDN_KERNELS,
    modeling_qwen3_5_moe: GDN_KERNELS,
    modeling_nemotron_h: (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "mamba2_split_conv1d_scan_combined",
        "mamba2_selective_state_update",
        "mamba2_chunk_scan",
    ),
}


def use_torch_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point each kernel global at the PyTorch function its kernel wrapper falls back to."""
    for module, names in KERNEL_GLOBALS.items():
        for name in names:
            monkeypatch.setattr(module, name, inspect.unwrap(getattr(module, name)))
