# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Distributed training: process group helpers and the actor shard runtime.

Import from this package:

``from agilerl.distributed import FSDPConfig, FSDPRuntime``

``process.py`` is rank / process group / collectives. ``fsdp_blocks.py`` is
``apply_fsdp2``; ``fsdp.py`` is ``CPUOffloadOptimizer`` and FSDP2 state
loading. ``FSDPConfig`` is defined torch-free in ``agilerl.arena.models.fsdp``
and re-exported here.
``runtime.py`` is ``BaseRuntime``.
"""

from .context_parallel import (
    all_to_all_hidden_to_seq,
    all_to_all_seq_to_hidden,
    apply_mixer_context_parallel,
    gather_for_cp,
    gather_for_cp_wo_grad,
    pad_right_for_cp,
    shard_for_cp,
    validate_cp_degree,
    validate_cp_heads,
    validate_cp_seq_len,
    validate_cp_style,
)
from .expert_parallel import (
    apply_expert_parallel,
    packed_expert_counts,
    reference_dispatch_combine,
    token_combine,
    token_dispatch,
    validate_actor_ep,
)
from .fsdp import (
    CPUOffloadOptimizer,
    FSDPConfig,
    full_shape_views,
    gather_params,
    materialize_dtensors,
    materialize_fsdp2_from_cpu_state,
    reshard_fsdp_modules,
)
from .fsdp_blocks import apply_fsdp2
from .mesh import (
    MeshAxisName,
    ParallelMesh,
    build_parallel_mesh,
    ep_data_parallel_size,
    tp_data_parallel_size,
    validate_ep_degree,
)
from .process import (
    aggregate_metrics_across_gpus,
    aggregate_metrics_dict,
    all_ranks,
    allreduce_minmax_int,
    allreduce_sum_ints,
    any_rank,
    barrier,
    broadcast_object_list,
    collective_device,
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
    "MeshAxisName",
    "ParallelMesh",
    "PrepareResult",
    "aggregate_metrics_across_gpus",
    "aggregate_metrics_dict",
    "all_ranks",
    "all_to_all_hidden_to_seq",
    "all_to_all_seq_to_hidden",
    "allreduce_minmax_int",
    "allreduce_sum_ints",
    "any_rank",
    "apply_expert_parallel",
    "apply_fsdp2",
    "apply_mixer_context_parallel",
    "barrier",
    "broadcast_object_list",
    "build_parallel_mesh",
    "collective_device",
    "distributed_env_present",
    "ep_data_parallel_size",
    "full_shape_views",
    "gather_for_cp",
    "gather_for_cp_wo_grad",
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
    "pad_right_for_cp",
    "raise_on_any_rank",
    "reference_dispatch_combine",
    "reshard_fsdp_modules",
    "resolve_device",
    "set_seed",
    "shard_for_cp",
    "sync_grads",
    "token_combine",
    "token_dispatch",
    "tp_data_parallel_size",
    "validate_actor_ep",
    "validate_cp_degree",
    "validate_cp_heads",
    "validate_cp_seq_len",
    "validate_cp_style",
    "validate_ep_degree",
]
