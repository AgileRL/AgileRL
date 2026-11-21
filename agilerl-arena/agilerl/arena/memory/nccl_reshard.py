# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Trainer refit buffers of a merged-LoRA ``nccl_reshard`` weight sync."""

from __future__ import annotations

import agilerl.arena.memory.lora_formulas as lora_formulas
from agilerl.arena.memory.breakdown import MemoryComponent
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    DeviceSpec,
    ModelArch,
    TrainingSettings,
)


def refit_buffer_component(
    arch: ModelArch,
    settings: TrainingSettings,
    device: DeviceSpec,
    engine_tensor_parallel: int,
) -> MemoryComponent:
    """The main rank's ``nccl_reshard`` refit buffers, the larger of its bulk and misc phases, and the refit's NCCL groups.

    :param arch: Model architecture.
    :param settings: Training settings of a merged-LoRA async run.
    :param device: Training GPU.
    :param engine_tensor_parallel: Engine width the bulk weights reshard into.
    :return: The ``lora_send_buffer`` component.
    """
    refit = lora_formulas.nccl_reshard_refit_bytes(
        arch,
        settings.lora_target_scope,
        settings.lora_rank,
        lora_formulas.NcclReshardLayout(
            1 if settings.fsdp is None else settings.shard_group_gpus,
            engine_tensor_parallel,
        ),
        DTYPE_BYTES[settings.weight_dtype],
    )
    # The bulk m2n group and the misc broadcast group each rank joins on top
    # of the trainer's own. On 8x H100 (Qwen3.5-2B, 9B, Qwen3.8-27B) a trainer
    # rank holds 1.53-1.79 GiB more non-torch memory than under merged NIXL;
    # two groups charge 1.86 GiB.
    communicators = 2 * device.nccl_group_bytes
    return MemoryComponent(
        key="lora_send_buffer",
        label="nccl_reshard refit buffers",
        n_bytes=refit.total + communicators,
        detail={
            "bulk_group": refit.bulk_group,
            "bulk_fp32_temp": refit.bulk_fp32_temp,
            "misc_bucket": refit.misc_bucket,
            "misc_module": refit.misc_module,
            "misc_fp32_copy": refit.misc_fp32_copy,
            "communicators": communicators,
        },
        note=(
            "Main rank, the larger of two phases: the largest bulk block's "
            "merged local shards with one fp32 merge temp, then the misc "
            "bucket with the largest misc module gathered whole and its "
            "fp32 merge copy. Plus the NCCL memory of the refit's bulk and "
            "misc groups, outside the caching allocator."
        ),
    )
