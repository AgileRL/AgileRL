# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Expert Parallel (EP) mesh, shard placement, and torch all-to-all helpers.

Stage 1 targets HF packed-expert MoE under FSDP2: experts are ``Shard(0)``'d
on an ``ep`` mesh dim, tokens move with torch all-to-all, and expert modules
use a reduced FSDP mesh (``dp_mod_ep``). Context Parallel is deferred
(``cp=1``); mesh naming follows Prime-RL so a later CP merge can add
``ep % cp == 0`` composition without renaming dims.

Hard invariants:

- ``ep > 1`` requires FSDP (``fsdp_config``).
- Never densify a full MoE replica on one GPU for train-step / attach under
  ``ep > 1`` — use ``to_local()`` expert slices and LoRA-sized gathers only.
- ``ep == 1`` keeps today's flat process-group FSDP path (no DeviceMesh).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

DeviceMesh: Any
DTensor: Any
Shard: Any
distribute_module: Any
distribute_tensor: Any
init_device_mesh: Any
try:
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
    from torch.distributed.tensor import (
        DTensor,
        Shard,
        distribute_module,
        distribute_tensor,
    )
except ImportError:  # pragma: no cover -- torch built without distributed tensor
    DeviceMesh = None
    DTensor = None
    Shard = None
    distribute_module = None
    distribute_tensor = None
    init_device_mesh = None


@dataclass(frozen=True)
class EpMeshLayout:
    """Prime-RL-shaped EP mesh sizes for stage 1 (``cp=1``, no PP / HSDP replicate).

    :param world_size: Global process count (``dp_shard`` when ``dp_replicate=1``).
    :param ep: Expert-parallel degree (``dp_shard_in_ep`` when ``cp=1``).
    :param dp_shard: Data-shard world (equals ``world_size`` in stage 1).
    :param dp_shard_mod_ep: FSDP mesh size for expert modules (``dp_shard // ep``).
    :param dp_shard_in_ep: Ranks that share one DP microbatch and shard experts.
    """

    world_size: int
    ep: int
    dp_shard: int
    dp_shard_mod_ep: int
    dp_shard_in_ep: int

    @property
    def ep_enabled(self) -> bool:
        return self.ep > 1


def compute_ep_mesh_layout(world_size: int, ep: int) -> EpMeshLayout:
    """Derive ``dp_shard_mod_ep`` / ``dp_shard_in_ep`` from ``world_size`` and ``ep``.

    Stage 1 assumes ``cp=1`` and ``dp_replicate=1``, so ``dp_shard == world_size``
    and ``dp_shard_in_ep == ep``.
    """
    if world_size < 1:
        msg = f"world_size must be >= 1, got {world_size}"
        raise ValueError(msg)
    if ep < 1:
        msg = f"ep must be >= 1, got {ep}"
        raise ValueError(msg)
    if world_size % ep != 0:
        msg = (
            f"world_size ({world_size}) must be divisible by ep ({ep}) "
            "for Expert Parallel."
        )
        raise ValueError(msg)
    dp_shard = world_size
    dp_shard_mod_ep = dp_shard // ep
    dp_shard_in_ep = ep
    return EpMeshLayout(
        world_size=world_size,
        ep=ep,
        dp_shard=dp_shard,
        dp_shard_mod_ep=dp_shard_mod_ep,
        dp_shard_in_ep=dp_shard_in_ep,
    )


def validate_ep_config(
    ep: int,
    *,
    fsdp_config: Any | None,
    world_size: int | None = None,
    num_experts: int | None = None,
    is_moe: bool | None = None,
    cp: int = 1,
) -> None:
    """Fail loud on unsupported EP configurations (including CP×EP compose)."""
    if not isinstance(ep, int) or isinstance(ep, bool):
        msg = f"ep must be an int, got {type(ep).__name__}"
        raise TypeError(msg)
    if ep < 1:
        msg = f"ep must be >= 1, got {ep}"
        raise ValueError(msg)
    if ep == 1:
        return
    if fsdp_config is None:
        msg = (
            "ep > 1 requires fsdp_config. Expert Parallel shards experts only; "
            "attention / router / embeddings still need FSDP to avoid a full "
            "non-expert replica on every GPU."
        )
        raise ValueError(msg)
    if is_moe is False:
        msg = (
            "ep > 1 requires a packed-expert MoE model recognized by "
            "agilerl.algorithms.core.llm_ops.moe_lora detectors."
        )
        raise ValueError(msg)
    if world_size is not None and world_size % ep != 0:
        msg = f"world_size ({world_size}) must be divisible by ep ({ep})."
        raise ValueError(msg)
    if cp > 1 and world_size is not None:
        from agilerl.utils.parallel_dims import compute_hybrid_layout

        compute_hybrid_layout(world_size, cp=cp, ep=ep)
    if num_experts is not None:
        if num_experts < 1:
            msg = f"num_experts must be >= 1, got {num_experts}"
            raise ValueError(msg)
        if num_experts % ep != 0:
            msg = (
                f"num_experts ({num_experts}) must be divisible by ep ({ep}) "
                "in stage 1 (uneven expert packs are unsupported)."
            )
            raise ValueError(msg)


@dataclass
class ExpertParallelMesh:
    """DeviceMesh views for EP + FSDP composition (``cp=1``).

    Named views match Prime-RL:

    - ``world``: ``(dp_shard_mod_ep, dp_shard_in_ep)``
    - ``ep``: flattened ``dp_shard_in_ep`` (expert ``Shard(0)`` + token A2A)
    - ``dp_mod_ep``: ``dp_shard_mod_ep`` (FSDP for expert modules)
    - ``hsdp``: flattened full shard world (FSDP for non-expert params)
    """

    layout: EpMeshLayout
    world: Any  # DeviceMesh
    ep: Any  # DeviceMesh
    dp_mod_ep: Any  # DeviceMesh
    hsdp: Any  # DeviceMesh

    @property
    def ep_group(self) -> dist.ProcessGroup:
        return self.ep.get_group()

    @property
    def ep_degree(self) -> int:
        return self.layout.ep


def build_expert_parallel_mesh(
    world_size: int | None = None,
    ep: int = 1,
    *,
    cp: int = 1,
    device_type: str | None = None,
) -> ExpertParallelMesh | Any | None:
    """Build EP DeviceMesh views, or ``None`` when ``ep == 1`` (flat PG path).

    When ``cp > 1``, delegates to
    :func:`agilerl.utils.parallel_dims.build_hybrid_parallel_mesh` so EP and CP
    share one Prime-RL-shaped world (``ep % cp == 0``).

    Requires an initialised process group whose ``world_size`` matches
    ``world_size`` (default: current process group size).
    """
    if ep <= 1:
        return None
    if cp > 1:
        from agilerl.utils.parallel_dims import build_hybrid_parallel_mesh

        return build_hybrid_parallel_mesh(
            world_size, cp=cp, ep=ep, device_type=device_type
        )
    if init_device_mesh is None or DeviceMesh is None:
        msg = "Expert Parallel requires torch.distributed DeviceMesh support."
        raise RuntimeError(msg)
    if not dist.is_available() or not dist.is_initialized():
        msg = (
            "Expert Parallel requires an initialised process group. Launch "
            "with torchrun (or set rendezvous env vars) before building the mesh."
        )
        raise RuntimeError(msg)
    if world_size is None:
        world_size = dist.get_world_size()
    layout = compute_ep_mesh_layout(world_size, ep)
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # Keep dp_shard_mod_ep even when it is 1 — expert FSDP still needs the dim
    # for mixed-precision units (Prime-RL).
    dims = [layout.dp_shard_mod_ep, layout.dp_shard_in_ep]
    names = ["dp_shard_mod_ep", "dp_shard_in_ep"]
    world = init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))

    # When dp_shard_mod_ep == 1 the world mesh is 1-D on dp_shard_in_ep only if
    # init_device_mesh drops size-1 dims — we always pass both dims above so
    # both names exist. Flatten EP and HSDP views for collectives / FSDP.
    ep_mesh = world["dp_shard_in_ep"]
    if ep_mesh.ndim > 1:
        ep_mesh = ep_mesh._flatten(mesh_dim_name="ep")
    elif ep_mesh.mesh_dim_names != ("ep",):
        # Single dim: rename for stable get_mesh("ep") callers.
        ep_mesh = world["dp_shard_in_ep"]

    dp_mod_ep = world["dp_shard_mod_ep"]
    hsdp = world._flatten(mesh_dim_name="hsdp") if world.ndim > 1 else world

    return ExpertParallelMesh(
        layout=layout,
        world=world,
        ep=ep_mesh,
        dp_mod_ep=dp_mod_ep,
        hsdp=hsdp,
    )


def _partition_experts_fn(
    _name: str, mod: nn.Module, device_mesh: Any
) -> None:
    """``Shard(0)`` every direct parameter on the EP mesh; stash the EP group."""
    assert distribute_tensor is not None and Shard is not None
    for param_name, param in list(mod.named_parameters(recurse=False)):
        if isinstance(param, DTensor):
            continue
        sharded = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
        mod.register_parameter(param_name, sharded)
    mod._ep_group = device_mesh.get_group()
    mod._ep_mesh = device_mesh
    mod._ep_degree = int(device_mesh.size())


def shard_experts_on_ep(module: nn.Module, ep_mesh: Any) -> nn.Module:
    """Place packed expert weights as ``DTensor`` ``Shard(0)`` over ``ep_mesh``.

    Each rank's ``to_local()`` view has ``num_experts // ep`` experts on dim 0.
    Sets ``module._ep_group`` / ``_ep_mesh`` / ``_ep_degree`` for dispatch.
    """
    if ep_mesh is None:
        return module
    if distribute_tensor is None or Shard is None:
        msg = "Expert Parallel requires torch.distributed.tensor support."
        raise RuntimeError(msg)
    if distribute_module is not None:
        return distribute_module(module, ep_mesh, partition_fn=_partition_experts_fn)
    _partition_experts_fn("", module, ep_mesh)
    return module


def expert_local_tensor(weight: torch.Tensor) -> torch.Tensor:
    """Dense local expert shard (``to_local`` for ``DTensor``, else identity)."""
    if DTensor is not None and isinstance(weight, DTensor):
        return weight.to_local()
    return weight


def expert_param_bytes_local(module: nn.Module) -> int:
    """Sum of local (sharded) expert parameter bytes on this rank."""
    total = 0
    for param in module.parameters(recurse=True):
        local = expert_local_tensor(param)
        total += local.numel() * local.element_size()
    return total


@dataclass
class TokenDispatchState:
    """Metadata to reverse an EP token all-to-all after local expert compute."""

    input_splits: list[int]
    output_splits: list[int]
    ep_group: dist.ProcessGroup
    ep_degree: int
    num_local_experts: int
    # After A2A, token rows are rank-major per local expert; permute gathers
    # rows into local-expert-major order for grouped GEMM.
    permute_indices: torch.Tensor | None = None
    # Counts in local-expert-major order: shape ``[num_local_experts]``.
    num_tokens_per_local_expert: torch.Tensor | None = None


def _all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Synchronous ``all_to_all_single`` returning ``output``."""
    dist.all_to_all_single(
        output,
        input.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


class _AllToAllVar(torch.autograd.Function):
    """Variable-split all-to-all with autograd (torch EP backend)."""

    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.group = group
        out_rows = sum(output_splits)
        output = input.new_empty((out_rows,) + tuple(input.shape[1:]))
        _all_to_all_single(output, input, output_splits, input_splits, group)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        grad_input = grad_output.new_empty(
            (sum(ctx.input_splits),) + tuple(grad_output.shape[1:])
        )
        _all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            ctx.input_splits,
            ctx.output_splits,
            ctx.group,
        )
        return grad_input, None, None, None


def all_to_all_single_autograd(
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Autograd-aware variable-split all-to-all over ``group``."""
    return _AllToAllVar.apply(input, output_splits, input_splits, group)


def _permute_to_local_expert_major(
    routed_input: torch.Tensor,
    num_tokens_per_expert_group: torch.Tensor,
    ep_degree: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder rank-major A2A output into local-expert-major token rows.

    ``num_tokens_per_expert_group`` layout from the count A2A is::

        [e0@r0, e1@r0, ..., e0@r1, e1@r1, ...]

    Grouped expert GEMMs want::

        [e0 from all ranks, e1 from all ranks, ...]
    """
    counts = num_tokens_per_expert_group.view(ep_degree, num_local_experts)
    # tokens per local expert across ranks
    local_counts = counts.sum(dim=0)
    # Build gather indices without host sync beyond the counts already synced.
    # Split sizes for the incoming buffer are row chunks in rank-major order.
    split_sizes = counts.reshape(-1).tolist()
    chunks = list(routed_input.split(split_sizes, dim=0))
    # chunks[r * num_local_experts + e] -> tokens for local expert e from rank r
    per_expert: list[torch.Tensor] = []
    index_parts: list[torch.Tensor] = []
    row_ids = torch.arange(routed_input.shape[0], device=routed_input.device)
    row_chunks = list(row_ids.split(split_sizes, dim=0))
    for expert in range(num_local_experts):
        expert_chunks = [
            chunks[rank * num_local_experts + expert] for rank in range(ep_degree)
        ]
        expert_rows = [
            row_chunks[rank * num_local_experts + expert] for rank in range(ep_degree)
        ]
        per_expert.append(
            torch.cat(expert_chunks, dim=0) if expert_chunks else routed_input[:0]
        )
        index_parts.append(
            torch.cat(expert_rows, dim=0) if expert_rows else row_ids[:0]
        )
    permuted = torch.cat(per_expert, dim=0) if per_expert else routed_input
    permute_indices = torch.cat(index_parts, dim=0) if index_parts else row_ids
    return permuted, permute_indices, local_counts


def _unpermute_from_local_expert_major(
    routed_output: torch.Tensor,
    permute_indices: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    """Inverse of :func:`_permute_to_local_expert_major` for combine A2A."""
    out = routed_output.new_empty((num_rows,) + tuple(routed_output.shape[1:]))
    out[permute_indices] = routed_output
    return out


def token_dispatch(
    routed_input: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    *,
    ep_group: dist.ProcessGroup,
    ep_degree: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, TokenDispatchState]:
    """All-to-all tokens to expert-owning ranks; return local-expert-major rows.

    ``routed_input`` must be ordered by **global** expert id with
    ``num_tokens_per_expert`` of shape ``[ep_degree * num_local_experts]``
    (tokens for expert ``g`` contiguous). Each rank sends tokens whose global
    expert maps to a remote EP peer and receives tokens for its local experts.
    """
    if num_tokens_per_expert.numel() != ep_degree * num_local_experts:
        msg = (
            f"num_tokens_per_expert length {num_tokens_per_expert.numel()} != "
            f"ep_degree * num_local_experts ({ep_degree} * {num_local_experts})"
        )
        raise ValueError(msg)

    with torch.no_grad():
        num_tokens_per_expert_group = torch.empty_like(num_tokens_per_expert)
        dist.all_to_all_single(
            num_tokens_per_expert_group,
            num_tokens_per_expert.contiguous(),
            group=ep_group,
        )
        input_splits = (
            num_tokens_per_expert.view(ep_degree, num_local_experts).sum(dim=1).tolist()
        )
        output_splits = (
            num_tokens_per_expert_group.view(ep_degree, num_local_experts)
            .sum(dim=1)
            .tolist()
        )

    dispatched = all_to_all_single_autograd(
        routed_input, output_splits, input_splits, ep_group
    )
    permuted, permute_indices, local_counts = _permute_to_local_expert_major(
        dispatched, num_tokens_per_expert_group, ep_degree, num_local_experts
    )
    state = TokenDispatchState(
        input_splits=input_splits,
        output_splits=output_splits,
        ep_group=ep_group,
        ep_degree=ep_degree,
        num_local_experts=num_local_experts,
        permute_indices=permute_indices,
        num_tokens_per_local_expert=local_counts,
    )
    return permuted, local_counts, state


def token_combine(
    routed_output: torch.Tensor,
    state: TokenDispatchState,
) -> torch.Tensor:
    """Reverse :func:`token_dispatch` — unpermute then all-to-all back."""
    assert state.permute_indices is not None
    num_rows = sum(state.output_splits)
    unpermuted = _unpermute_from_local_expert_major(
        routed_output, state.permute_indices, num_rows
    )
    # Combine swaps splits relative to dispatch.
    return all_to_all_single_autograd(
        unpermuted,
        state.input_splits,
        state.output_splits,
        state.ep_group,
    )


# Locked attach / parallelize order when ``ep > 1`` (Phase 1):
# 1. Dense actor on CPU (or meta) — never ``model.to(cuda)`` before shard.
# 2. PEFT LoRA attach on CPU (full-E bookkeeping is fine on host).
# 3. ``upgrade_moe_param_wrappers`` on the CPU module.
# 4. ``apply_expert_parallel`` — ``Shard(0)`` packed experts + expert LoRA on ``ep``.
# 5. FSDP2 via ``materialize_fsdp2_from_cpu_state``: experts on ``dp_mod_ep``,
#    non-experts on ``hsdp`` (Phase 3 mesh kwargs).
# Forbidden: CUDA full-expert gather around PEFT attach / upgrade under ``ep > 1``.
EP_ATTACH_ORDER = (
    "cpu_dense",
    "peft_attach",
    "upgrade_moe_wrappers",
    "ep_shard",
    "fsdp2_materialize",
)


def module_ep_degree(module: nn.Module) -> int:
    """EP degree stashed by :func:`shard_experts_on_ep`, else ``1``."""
    return int(getattr(module, "_ep_degree", 1) or 1)


def module_ep_group(module: nn.Module) -> dist.ProcessGroup | None:
    """EP process group stashed by :func:`shard_experts_on_ep`, if any."""
    return getattr(module, "_ep_group", None)


def iter_packed_expert_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Named modules that match ``moe_lora`` packed-expert detectors."""
    from agilerl.algorithms.core.llm_ops.moe_lora import (
        _is_routed_experts_module,
        _is_sorted_experts_module,
    )

    found: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if _is_sorted_experts_module(module) or _is_routed_experts_module(module):
            found.append((name, module))
    return found


def _shard_lora_linear_on_ep(
    linear: nn.Module,
    ep_mesh: Any,
    *,
    expert_dim: int,
    lora_rank: int | None = None,
    num_experts: int | None = None,
) -> None:
    """Shard a PEFT LoRA Linear on the stacked expert axis.

    ``A`` is expert-major ``[E*r, in]`` — contiguous ``Shard(0)``. ``B`` is PEFT
    ``[out, E*r]`` packed rank-major as ``[out, r, E]``; shard that last axis
    (``Shard(2)``) so each rank owns a contiguous expert block.
    """
    if distribute_tensor is None or Shard is None:
        msg = "Expert Parallel requires torch.distributed.tensor support."
        raise RuntimeError(msg)
    weight = getattr(linear, "weight", None)
    if weight is None or isinstance(weight, DTensor):
        return
    if expert_dim == 1:
        # PEFT B packs ``[out, E*r]`` as rank-major ``view(out, r, E)``.
        if weight.ndim != 2:
            msg = (
                f"Expected 2D LoRA B weight before EP reshape, got shape "
                f"{tuple(weight.shape)}"
            )
            raise ValueError(msg)
        if lora_rank is None or num_experts is None:
            msg = "LoRA B EP shard requires lora_rank and num_experts."
            raise ValueError(msg)
        out_f, stacked = weight.shape
        if stacked != lora_rank * num_experts:
            msg = (
                f"LoRA B shape {tuple(weight.shape)} is not "
                f"[out, E*r] with E={num_experts}, r={lora_rank}."
            )
            raise ValueError(msg)
        if num_experts % ep_mesh.size() != 0:
            msg = (
                f"LoRA B expert count {num_experts} must be divisible by "
                f"ep ({ep_mesh.size()})."
            )
            raise ValueError(msg)
        weight_3d = (
            weight.detach()
            .reshape(out_f, lora_rank, num_experts)
            .contiguous()
            .clone()
        )
        sharded = distribute_tensor(weight_3d, ep_mesh, [Shard(2)])
    else:
        if weight.ndim != 2:
            msg = (
                f"Expected 2D LoRA weight for EP shard, got shape "
                f"{tuple(weight.shape)}"
            )
            raise ValueError(msg)
        if weight.shape[expert_dim] % ep_mesh.size() != 0:
            msg = (
                f"LoRA weight dim {expert_dim} size {weight.shape[expert_dim]} "
                f"must be divisible by ep ({ep_mesh.size()})."
            )
            raise ValueError(msg)
        sharded = distribute_tensor(weight, ep_mesh, [Shard(expert_dim)])
    linear.register_parameter("weight", nn.Parameter(sharded))
    linear._ep_group = ep_mesh.get_group()
    linear._ep_mesh = ep_mesh
    linear._ep_degree = int(ep_mesh.size())


def install_ep_routed_forward(module: nn.Module) -> bool:
    """Point packed routed-experts ``forward`` at the EP-aware local GEMM path.

    Needed when ``ep > 1`` without PEFT ``RoutedExpertsLoraWrapper`` (attention-
    only LoRA). Split-LoRA wrappers already call
    ``_routed_experts_local_forward``; this installs the same path on the base
    HF experts module. Idempotent.
    """
    import types

    from agilerl.algorithms.core.llm_ops.moe_lora import (
        _is_routed_experts_module,
        _routed_experts_local_forward,
    )

    if getattr(module, "_agilerl_ep_forward", False):
        return False
    if not _is_routed_experts_module(module):
        return False

    def _ep_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return _routed_experts_local_forward(
            self, hidden_states, top_k_index, top_k_weights
        )

    module.forward = types.MethodType(_ep_forward, module)
    module._agilerl_ep_forward = True
    return True


def apply_expert_parallel(model: nn.Module, ep_mesh: Any) -> int:
    """``Shard(0)`` packed-expert base weights on ``ep_mesh``; shard expert LoRA too.

    PEFT stacked LoRA ``A`` is expert-major ``[E*r, in]`` (``Shard(0)``). ``B``
    is ``[out, E*r]`` viewed as ``[out, r, E]`` and ``Shard(2)``'d on the expert
    axis. Also installs an EP-aware ``forward`` on routed packed-expert modules
    so non-LoRA / attention-only LoRA paths still dispatch tokens. Returns how
    many packed-expert base modules were parallelized. No-op when ``ep_mesh``
    is ``None``. Does not densify full expert replicas on CUDA.
    """
    if ep_mesh is None:
        return 0
    count = 0
    for _name, module in iter_packed_expert_modules(model):
        shard_experts_on_ep(module, ep_mesh)
        install_ep_routed_forward(module)
        count += 1
        parent_wrappers = [
            m
            for m in model.modules()
            if getattr(m, "base_layer", None) is module
            or (
                hasattr(m, "get_base_layer")
                and callable(m.get_base_layer)
                and m.get_base_layer() is module
            )
        ]
        for wrapper in parent_wrappers:
            lora_a = getattr(wrapper, "lora_A", None)
            lora_b = getattr(wrapper, "lora_B", None)
            num_experts = int(getattr(wrapper, "num_experts", 0) or 0)
            if isinstance(lora_a, nn.ModuleDict):
                for adapter in lora_a.values():
                    _shard_lora_linear_on_ep(adapter, ep_mesh, expert_dim=0)
            if isinstance(lora_b, nn.ModuleDict):
                for adapter_name, adapter in lora_b.items():
                    rank = int(wrapper.r[adapter_name])
                    _shard_lora_linear_on_ep(
                        adapter,
                        ep_mesh,
                        expert_dim=1,
                        lora_rank=rank,
                        num_experts=num_experts,
                    )
    return count


def global_expert_owned_by_rank(
    expert_id: int, *, ep_degree: int, num_local_experts: int
) -> int:
    """EP rank that owns global expert ``expert_id`` (contiguous block mapping)."""
    return int(expert_id) // num_local_experts


def local_expert_index(expert_id: int, *, num_local_experts: int) -> int:
    """Map a global expert id to the owning rank's local index."""
    return int(expert_id) % num_local_experts


def reference_dispatch_combine(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    ep_degree: int,
    num_experts: int,
) -> torch.Tensor:
    """Single-process reference: identity round-trip for dispatch/combine ordering.

    Sorts tokens by global expert, splits into EP-rank ownership blocks, and
    rebuilds the original order — used to validate multi-rank A2A against a
    deterministic permute when each "rank" applies a local identity expert map.
    """
    if num_experts % ep_degree != 0:
        msg = f"num_experts ({num_experts}) must be divisible by ep ({ep_degree})"
        raise ValueError(msg)
    num_local = num_experts // ep_degree
    order = torch.argsort(expert_ids, stable=True)
    sorted_tokens = tokens[order]
    counts = torch.bincount(expert_ids, minlength=num_experts)
    # Simulate per-rank receive buffers then expert-major permute + unpermute.
    # Identity expert map → combine restores sorted order; inverse argsort restores input.
    pieces: list[torch.Tensor] = []
    for rank in range(ep_degree):
        start = rank * num_local
        end = start + num_local
        local_counts = counts[start:end]
        # tokens for this rank's experts in expert-major order already via global sort
        offsets = [int(counts[:start].sum().item()), int(counts[:end].sum().item())]
        pieces.append(sorted_tokens[offsets[0] : offsets[1]])
        _ = local_counts  # documented for readers matching token_dispatch
    restored_sorted = torch.cat(pieces, dim=0) if pieces else sorted_tokens
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return restored_sorted[inverse]
