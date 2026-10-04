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

from .expert_parallel import (
    ParallelMesh,
    apply_expert_parallel,
    build_parallel_mesh,
    ep_data_parallel_size,
    packed_expert_counts,
    reference_dispatch_combine,
    token_combine,
    token_dispatch,
    tp_data_parallel_size,
    validate_actor_ep,
    validate_ep_degree,
)
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
    "ParallelMesh",
    "PrepareResult",
    "aggregate_metrics_across_gpus",
    "aggregate_metrics_dict",
    "all_ranks",
    "allreduce_minmax_int",
    "any_rank",
    "apply_expert_parallel",
    "apply_fsdp2",
    "barrier",
    "broadcast_object_list",
    "build_parallel_mesh",
    "distributed_env_present",
    "ep_data_parallel_size",
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
    "packed_expert_counts",
    "raise_on_any_rank",
    "reference_dispatch_combine",
    "reshard_fsdp_modules",
    "resolve_device",
    "set_seed",
    "sync_grads",
    "token_combine",
    "token_dispatch",
    "tp_data_parallel_size",
    "validate_actor_ep",
    "validate_ep_degree",
]
