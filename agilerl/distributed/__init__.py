# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Distributed training: process group helpers and the actor shard runtime.

Import from this package:

``from agilerl.distributed import FSDPConfig, FSDPRuntime``

``process.py`` is rank / process group / collectives. ``fsdp.py`` is
``apply_fsdp2`` / ``CPUOffloadOptimizer``; ``FSDPConfig`` is defined
torch-free in ``agilerl.arena.models.fsdp`` and re-exported here.
``runtime.py`` is ``BaseRuntime``.
"""

from .fsdp import (
    CPUOffloadOptimizer,
    FSDPConfig,
    apply_fsdp2,
    full_shape_views,
    gather_params,
    materialize_dtensors,
    materialize_fsdp2_from_cpu_state,
    reshard_fsdp_modules,
)
from .process import (
    aggregate_metrics_across_gpus,
    aggregate_metrics_dict,
    all_ranks,
    allreduce_minmax_int,
    any_rank,
    barrier,
    broadcast_object_list,
    distributed_env_present,
    gather_objects,
    gather_tensor,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    raise_on_any_rank,
    resolve_device,
    set_seed,
    sync_grads,
)
from .runtime import (
    BaseRuntime,
    DPRuntime,
    FSDPRuntime,
    PrepareResult,
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
    "is_main_process",
    "materialize_dtensors",
    "materialize_fsdp2_from_cpu_state",
    "raise_on_any_rank",
    "reshard_fsdp_modules",
    "resolve_device",
    "set_seed",
    "sync_grads",
]
