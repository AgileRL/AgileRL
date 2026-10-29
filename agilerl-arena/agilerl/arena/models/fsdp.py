# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""FSDP2 shard settings for LLM training (torch-free)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

from pydantic import ConfigDict, Field

FSDP_DTYPE_NAMES = frozenset({"bfloat16", "float16", "float32", "float64"})
# Parameters with fewer elements stay unsharded on every rank.
PARAM_PERSISTENCE_THRESHOLD = 100_000
# FSDP units all-gathered ahead of the running one, in forward and in backward.
PREFETCH_UNITS = 1


@dataclass
class FSDPConfig:
    """Settings for sharding the LLM actor with PyTorch FSDP2 (``fully_shard``)."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    reshard_after_forward: Annotated[
        bool | None,
        Field(
            description=(
                "Free gathered parameters after each module's forward. False "
                "keeps them gathered: faster, more VRAM. Picked from the memory "
                "estimate; manifests cannot set it. None picks False when the "
                "estimated training peak fits with the underprediction buffer "
                "at every other fastest choice, else True; off CUDA it picks True."
            ),
        ),
    ] = None
    cpu_offload: Annotated[
        bool | None,
        Field(
            description=(
                "Offload sharded parameters and gradients to CPU. Needs "
                "colocated vLLM generation: no async rollout engines. Mutually "
                "exclusive with optim_cpu_offload. Picked from the memory "
                "estimate; manifests cannot set it. None picks True only when "
                "vLLM is colocated and the training peak is over the GPU at "
                "every other lightest choice, activation offload and the "
                "micro-batch floor; off CUDA it picks False."
            ),
        ),
    ] = None
    optim_cpu_offload: Annotated[
        bool | None,
        Field(
            description=(
                "Keep Adam state on CPU and step it there with the default "
                "AdamW kernel. Parameters and gradients stay on GPU. False "
                "keeps Adam on GPU and steps it with fused AdamW. Picked from "
                "the memory estimate; manifests cannot set it. None picks "
                "GPU when the estimated training peak fits with the "
                "underprediction buffer, else CPU; off CUDA it picks CPU."
            ),
        ),
    ] = None
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
    routed_expert_chunk_mib: Annotated[
        int | None,
        Field(
            ge=1,
            description=(
                "MiB of the widest activation of one routed-expert row chunk "
                "under expert LoRA, of one fp32 chunk of the expert-parallel "
                "combine, and of one grouped LoRA GEMM output. Bigger chunks "
                "launch fewer kernels per MoE layer; on Super-VL the learn "
                "peak grows ~0.37 GiB per GiB of chunk. Picked from the memory "
                "estimate; manifests cannot set it. None picks the "
                "largest of 512/256/128/64 whose estimated training peak "
                "fits, else 64; off CUDA it picks 64."
            ),
        ),
    ] = None
    activation_offload: Annotated[
        bool | None,
        Field(
            description=(
                "Offload backward-saved activations to host memory between "
                "forward and backward. Saves GPU memory at the cost of PCIe "
                "traffic. Picked from the memory estimate; manifests cannot "
                "set it. None picks True only when the training peak is over "
                "the GPU at every other lightest choice and a micro-batch of "
                "1; off CUDA it picks False."
            ),
        ),
    ] = None
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

    def __post_init__(self) -> None:
        self.param_dtype = _dtype_name(self.param_dtype)
        self.reduce_dtype = _dtype_name(self.reduce_dtype)
        if self.ep < 1:
            msg = "FSDPConfig.ep must be >= 1"
            raise ValueError(msg)
        if self.ep_token_blocks < 1:
            msg = "FSDPConfig.ep_token_blocks must be >= 1"
            raise ValueError(msg)
        if (
            self.routed_expert_chunk_mib is not None
            and self.routed_expert_chunk_mib < 1
        ):
            msg = "FSDPConfig.routed_expert_chunk_mib must be >= 1"
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
