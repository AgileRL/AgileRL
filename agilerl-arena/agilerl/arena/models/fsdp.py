# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""FSDP2 shard settings for LLM training (torch-free)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

from pydantic import ConfigDict, Field

FSDP_DTYPE_NAMES = frozenset({"bfloat16", "float16", "float32", "float64"})


@dataclass
class FSDPConfig:
    """Settings for sharding the LLM actor with PyTorch FSDP2 (``fully_shard``)."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    reshard_after_forward: Annotated[
        bool,
        Field(
            description=(
                "Free gathered parameters after each module's forward. False "
                "keeps them gathered: faster, more VRAM."
            ),
        ),
    ] = True
    cpu_offload: Annotated[
        bool,
        Field(
            description=(
                "Offload sharded parameters and gradients to CPU. Needs "
                "colocated vLLM. Mutually exclusive with optim_cpu_offload."
            ),
        ),
    ] = False
    optim_cpu_offload: Annotated[
        bool,
        Field(
            description=(
                "Keep Adam state on CPU and move it to GPU only for step(). "
                "Parameters and gradients stay on GPU."
            ),
        ),
    ] = True
    defer_grad_sync: Annotated[
        bool,
        Field(
            description=(
                "Reduce-scatter gradients only on the last micro-batch of an "
                "optimizer step. Saves communication; holds unsharded grads "
                "in between."
            ),
        ),
    ] = True
    param_dtype: Annotated[
        str,
        Field(
            description=(
                "Mixed-precision parameter dtype, e.g. 'bfloat16'. A torch.dtype "
                "is accepted and stored by name."
            ),
        ),
    ] = "bfloat16"
    reduce_dtype: Annotated[
        str,
        Field(description="Dtype for gradient reduce-scatter and all-reduce."),
    ] = "float32"
    prefetch_units: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Neighbouring FSDP units to all-gather ahead during forward and "
                "backward. 1 gathers unit i+1 while unit i runs."
            ),
        ),
    ] = 1
    wrap_every_n_blocks: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Consecutive transformer blocks per FSDP unit. Larger values "
                "mean fewer, bigger collectives."
            ),
        ),
    ] = 1
    param_persistence_threshold: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "Parameters with fewer elements than this stay unsharded. "
                "0 shards every parameter."
            ),
        ),
    ] = 100_000

    def __post_init__(self) -> None:
        self.param_dtype = _dtype_name(self.param_dtype)
        self.reduce_dtype = _dtype_name(self.reduce_dtype)
        if self.prefetch_units < 1:
            msg = "FSDPConfig.prefetch_units must be >= 1"
            raise ValueError(msg)
        if self.wrap_every_n_blocks < 1:
            msg = "FSDPConfig.wrap_every_n_blocks must be >= 1"
            raise ValueError(msg)
        if self.param_persistence_threshold < 0:
            msg = "FSDPConfig.param_persistence_threshold must be >= 0"
            raise ValueError(msg)


def _dtype_name(value: object) -> str:
    """Normalize ``bfloat16`` / ``torch.bfloat16`` / a ``torch.dtype`` to its name."""
    name = str(value).removeprefix("torch.")
    if name not in FSDP_DTYPE_NAMES:
        msg = (
            f"Unknown FSDP dtype {value!r}; expected one of {sorted(FSDP_DTYPE_NAMES)}"
        )
        raise ValueError(msg)
    return name
