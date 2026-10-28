# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""NCCL process-group bytes for one trainer rank."""

from __future__ import annotations

from agilerl.arena.memory.specs import MiB

# NCCL bytes as fractions of the device's 4-rank on-node group bytes, with
# NCCL cuMem off. A group whose ranks share a node holds NVLink buffers that
# grow with its size; a group across nodes holds few socket buffers. Each
# multi-rank trainer also pays a fixed base once.
NCCL_BASE_FRACTION = 0.14
NCCL_CROSS_NODE_FRACTION = 0.06
# All-to-all buffers expert parallel's own group holds beside its group bytes,
# on one node and across nodes.
NCCL_ALL_TO_ALL_ON_NODE_FRACTION = 1.1
NCCL_ALL_TO_ALL_CROSS_NODE_FRACTION = 0.54
# Non-torch memory a tensor-parallel trainer holds beyond its process groups.
TP_RUNTIME_BYTES = 512 * MiB


def nccl_on_node_fraction(group_size: int) -> float:
    """On-node group bytes against a 4-rank group, by group size."""
    if group_size <= 2:
        return 0.65
    if group_size <= 4:
        return 1.0
    return 1.13


def trainer_nccl_bytes(
    group_bytes: int,
    n_gpus: int,
    shard_gpus: int,
    expert_parallel: int = 1,
    tensor_parallel: int = 1,
    gpus_per_node: int | None = None,
) -> int:
    """NCCL buffers on one trainer rank, summed over the process groups it joins.

    Ranks fill nodes in rank order. The world group joins every rank. HSDP
    adds its shard group (consecutive ranks) and its replicate group (one
    rank per shard). EP and TP each add their own group of consecutive ranks,
    plus the group of ranks holding the same shard when that is more than one
    rank (two groups under HSDP: the split and its flattened replicas). EP's
    own group also holds all-to-all buffers.

    :param group_bytes: The device's 4-rank on-node group bytes.
    :param gpus_per_node: GPUs on one node; ``None`` keeps every rank on one node.
    """
    node = n_gpus if gpus_per_node is None else gpus_per_node

    def group(size: int, span: int) -> float:
        if span <= node:
            return nccl_on_node_fraction(size)
        return NCCL_CROSS_NODE_FRACTION

    replicate = n_gpus // shard_gpus
    fraction = NCCL_BASE_FRACTION + group(n_gpus, n_gpus)
    if replicate > 1:
        fraction += group(shard_gpus, shard_gpus) + group(replicate, n_gpus)
    all_to_all = (
        NCCL_ALL_TO_ALL_ON_NODE_FRACTION
        if expert_parallel <= node
        else NCCL_ALL_TO_ALL_CROSS_NODE_FRACTION
    )
    for degree, own_group in (
        (expert_parallel, group(expert_parallel, expert_parallel) + all_to_all),
        (tensor_parallel, group(tensor_parallel, tensor_parallel)),
    ):
        if degree <= 1:
            continue
        fraction += own_group
        outer = shard_gpus // degree
        if outer > 1:
            fraction += group(outer, shard_gpus)
            if replicate > 1:
                fraction += group(outer * replicate, n_gpus)
    return int(group_bytes * fraction)
