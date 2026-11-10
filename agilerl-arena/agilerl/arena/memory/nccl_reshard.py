# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Trainer refit buffers of a merged-LoRA ``nccl_reshard`` weight sync."""

from __future__ import annotations

import agilerl.arena.memory.lora_formulas as lora_formulas
from agilerl.arena.memory.breakdown import MemoryComponent
from agilerl.arena.memory.specs import DTYPE_BYTES, ModelArch, TrainingSettings


def refit_buffer_component(
    arch: ModelArch, settings: TrainingSettings, engine_tensor_parallel: int
) -> MemoryComponent:
    """The main rank's ``nccl_reshard`` refit buffers, the larger of its bulk and misc phases.

    :param arch: Model architecture.
    :param settings: Training settings of a merged-LoRA async run.
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
    return MemoryComponent(
        key="lora_send_buffer",
        label="nccl_reshard refit buffers",
        n_bytes=refit.total,
        detail={
            "bulk_group": refit.bulk_group,
            "bulk_fp32_temp": refit.bulk_fp32_temp,
            "misc_bucket": refit.misc_bucket,
            "misc_module": refit.misc_module,
            "misc_fp32_copy": refit.misc_fp32_copy,
        },
        note=(
            "Main rank, the larger of two phases: the largest bulk block's "
            "merged local shards with one fp32 merge temp, then the misc "
            "bucket with the largest misc module gathered whole and its "
            "fp32 merge copy."
        ),
    )
