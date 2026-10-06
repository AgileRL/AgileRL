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
                "Neighbouring FSDP units to all-gather ahead during forward. "
                "1 gathers unit i+1 while unit i runs."
            ),
        ),
    ] = 1
    backward_prefetch_units: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Neighbouring FSDP units to all-gather ahead during backward. "
                "1 gathers unit i-1 while unit i runs backward."
            ),
        ),
    ] = 1
    checkpoint_skip_layer_types: Annotated[
        tuple[str, ...],
        Field(
            description=(
                "Transformer block kinds left out of activation checkpointing. "
                "A block's kind is its block_type (hybrid models, e.g. "
                "'linear_attention', 'full_attention', 'moe') or else its class "
                "name. Empty checkpoints every block. Applies when "
                "gradient_checkpointing is on."
            ),
        ),
    ] = ()
    checkpoint_every_n_blocks: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Checkpoint one in every n blocks not skipped by "
                "checkpoint_skip_layer_types. 1 checkpoints all of them."
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
    ep: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Expert-parallel degree: packed MoE experts per layer are "
                "split across this many GPUs. 1 keeps data parallel plus "
                "FSDP sharding with no expert split."
            ),
        ),
    ] = 1
    ep_token_blocks: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Token blocks per routed MoE layer under expert parallel. "
                "Each block runs dispatch, experts, and combine; on GPU the "
                "next block's all-to-all overlaps this block's experts. 1 "
                "moves every token in one all-to-all."
            ),
        ),
    ] = 1
    tp: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Tensor-parallel degree for dense layers. Ranks in one TP "
                "group share a batch shard. 1 gives every rank its own batch."
            ),
        ),
    ] = 1
    shard_group_size: Annotated[
        int | None,
        Field(
            ge=1,
            description=(
                "Ranks that shard weights between them; groups replicate "
                "(HSDP). Set to GPUs per node so weight gathers stay "
                "inside a node. None shards across all trainer ranks."
            ),
        ),
    ] = None
    compile_blocks: Annotated[
        bool,
        Field(
            description=(
                "torch.compile the dense submodules of each transformer block "
                "(norms, MLPs) in place. MoE experts, routers, attention and "
                "Mamba mixers stay eager."
            ),
        ),
    ] = False
    compile_backend: Annotated[
        str,
        Field(description="torch.compile backend for compile_blocks."),
    ] = "inductor"

    def __post_init__(self) -> None:
        self.param_dtype = _dtype_name(self.param_dtype)
        self.reduce_dtype = _dtype_name(self.reduce_dtype)
        self.checkpoint_skip_layer_types = tuple(self.checkpoint_skip_layer_types)
        if self.prefetch_units < 1:
            msg = "FSDPConfig.prefetch_units must be >= 1"
            raise ValueError(msg)
        if self.backward_prefetch_units < 1:
            msg = "FSDPConfig.backward_prefetch_units must be >= 1"
            raise ValueError(msg)
        if self.checkpoint_every_n_blocks < 1:
            msg = "FSDPConfig.checkpoint_every_n_blocks must be >= 1"
            raise ValueError(msg)
        if self.wrap_every_n_blocks < 1:
            msg = "FSDPConfig.wrap_every_n_blocks must be >= 1"
            raise ValueError(msg)
        if self.param_persistence_threshold < 0:
            msg = "FSDPConfig.param_persistence_threshold must be >= 0"
            raise ValueError(msg)
        if self.ep < 1:
            msg = "FSDPConfig.ep must be >= 1"
            raise ValueError(msg)
        if self.ep_token_blocks < 1:
            msg = "FSDPConfig.ep_token_blocks must be >= 1"
            raise ValueError(msg)
        if self.tp < 1:
            msg = "FSDPConfig.tp must be >= 1"
            raise ValueError(msg)
        if self.shard_group_size is not None:
            if self.shard_group_size < 1:
                msg = "FSDPConfig.shard_group_size must be >= 1"
                raise ValueError(msg)
            for name, degree in (("ep", self.ep), ("tp", self.tp)):
                if self.shard_group_size % degree:
                    msg = (
                        f"FSDPConfig.shard_group_size={self.shard_group_size} "
                        f"must be divisible by {name}={degree}"
                    )
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
