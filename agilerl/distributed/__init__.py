# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Distributed training: process group helpers and the actor shard runtime.

Import from this package:

``from agilerl.distributed import FSDPConfig, make_shard_runtime``

``process.py`` is rank / process group / ``FSDPConfig`` / ``apply_fsdp2`` /
``CPUOffloadOptimizer``. ``runtime.py`` is ``BaseRuntime``.
"""

from .process import (
    CPUOffloadOptimizer,
    FSDPConfig,
    aggregate_metrics_across_gpus,
    aggregate_metrics_dict,
    all_reduce_mean,
    all_ranks,
    allreduce_minmax_int,
    any_rank,
    apply_fsdp2,
    barrier,
    broadcast_object_list,
    distributed_env_present,
    full_shape_views,
    gather_objects,
    gather_params,
    gather_tensor,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_fsdp_sharded,
    is_main_process,
    materialize_dtensors,
    materialize_fsdp2_from_cpu_state,
    raise_on_any_rank,
    resolve_device,
    set_seed,
    shard_dataloader_kwargs,
    sync_grads,
)
from .runtime import (
    BaseRuntime,
    DPRuntime,
    FSDPRuntime,
    PrepareResult,
    make_shard_runtime,
)

__all__ = [
    "BaseRuntime",
    "CPUOffloadOptimizer",
    "DPRuntime",
    "FSDPConfig",
    "FSDPRuntime",
    "PrepareResult",
    "aggregate_metrics_across_gpus",
    "aggregate_metrics_dict",
    "all_reduce_mean",
    "all_ranks",
    "allreduce_minmax_int",
    "any_rank",
    "apply_fsdp2",
    "barrier",
    "broadcast_object_list",
    "distributed_env_present",
    "full_shape_views",
    "gather_objects",
    "gather_params",
    "gather_tensor",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_distributed",
    "is_fsdp_sharded",
    "is_main_process",
    "make_shard_runtime",
    "materialize_dtensors",
    "materialize_fsdp2_from_cpu_state",
    "raise_on_any_rank",
    "resolve_device",
    "set_seed",
    "shard_dataloader_kwargs",
    "sync_grads",
]
