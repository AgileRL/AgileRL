# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Nemotron-H Mamba2 mixers, attention, and MLPs; Granite 4 reuses the mixer.

Each rank runs a mixer over ``num_heads/tp`` heads and ``n_groups/tp`` groups,
then all-reduces the output. ``in_proj`` and ``conv1d`` rows are grouped per
rank as that rank's gate, conv channels, and dt so a ``Shard(0)`` chunk is one
local mixer. ``in_proj`` LoRA B is sharded in the same row order.

With fewer SSM groups than ``tp``, ranks share a group: each stores ``1/tp`` of
the B and C rows of ``in_proj`` and its LoRA B and all-gathers them in forward.
A gated norm that spans every head sums its variance over the tensor-parallel
group, and the mixer then runs the unfused scan, since the fused kernel
normalizes its local heads.

The input and output pass through the tensor-parallel copy and reduce
functions, so gradients match the unsharded mixer. Attention, MLP, latent MoE
projection, and Omni ``vision_model`` blocks use the column- and row-parallel
layers in :mod:`agilerl.distributed.tensor_parallel`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import MethodType
from typing import Literal, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.func import functional_call
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridMambaLayer,
)
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHAttention,
    NemotronHMamba2Mixer,
    NemotronHMLP,
)

from agilerl.architectures.nemotron_h.mamba import mark_mamba_tp_shard
from agilerl.architectures.runtime import TensorParallelPlan
from agilerl.distributed.expert_parallel.dispatch import keep_collectives_eager
from agilerl.distributed.tensor_parallel import (
    ReduceFromTPRegion,
    SumOverTPRegion,
    add_row_bias,
    base_linear,
    base_weight,
    copy_input_to_region,
    gather_rows_once,
    install_gather_forward,
    install_reduce_forward,
    install_split_forward,
    is_linear,
    is_lora_b_weight,
    keep_lora_bank,
    local_region_params,
    on_tp_mesh,
    permute_rows,
    require_divisible,
    shard_attention_projections,
    shard_column,
    shard_dim0,
    shard_lora_b_rows,
    shard_reordered_rows,
    shard_row,
    share_lora_dropout,
    stash_tp,
    tp_degree,
)

MambaProjKind = Literal["in_proj", "conv"]
IsMixer = Callable[[nn.Module], bool]
Mamba2Mixer = NemotronHMamba2Mixer | GraniteMoeHybridMambaLayer


def is_nemotron_h_mixer(module: nn.Module) -> bool:
    """Whether ``module`` is a Nemotron-H Mamba2 mixer."""
    return isinstance(module, NemotronHMamba2Mixer)


def _mixer_dims(mixer: Mamba2Mixer) -> tuple[int, int, int, int]:
    # Hugging Face types these config-backed sizes as optional.
    return (
        cast("int", mixer.num_heads),
        cast("int", mixer.head_dim),
        cast("int", mixer.n_groups),
        cast("int", mixer.ssm_state_size),
    )


def _shares_groups(n_groups: int, tp: int) -> bool:
    """Whether several TP ranks read one SSM group's B and C rows."""
    return n_groups % tp != 0


def _rank_row_ranges(
    num_heads: int,
    head_dim: int,
    n_groups: int,
    ssm_state_size: int,
    tp: int,
    rank: int,
    kind: MambaProjKind,
) -> list[tuple[int, int]]:
    """Rows of one rank's gate/x, B, C, and dt inside the full projection."""
    intermediate = num_heads * head_dim
    local_heads = num_heads // tp
    local_groups = max(n_groups // tp, 1)
    head0 = rank * local_heads
    group0 = head0 // (num_heads // n_groups)
    head_span = local_heads * head_dim
    group_span = local_groups * ssm_state_size
    if kind == "conv":
        x0 = 0
        b0 = intermediate
        c0 = intermediate + n_groups * ssm_state_size
        return [
            (x0 + head0 * head_dim, x0 + head0 * head_dim + head_span),
            (b0 + group0 * ssm_state_size, b0 + group0 * ssm_state_size + group_span),
            (c0 + group0 * ssm_state_size, c0 + group0 * ssm_state_size + group_span),
        ]
    gate0 = 0
    x0 = intermediate
    b0 = 2 * intermediate
    c0 = 2 * intermediate + n_groups * ssm_state_size
    dt0 = c0 + n_groups * ssm_state_size
    return [
        (gate0 + head0 * head_dim, gate0 + head0 * head_dim + head_span),
        (x0 + head0 * head_dim, x0 + head0 * head_dim + head_span),
        (b0 + group0 * ssm_state_size, b0 + group0 * ssm_state_size + group_span),
        (c0 + group0 * ssm_state_size, c0 + group0 * ssm_state_size + group_span),
        (dt0 + head0, dt0 + head0 + local_heads),
    ]


def _stored_row_ranges(
    num_heads: int,
    head_dim: int,
    n_groups: int,
    ssm_state_size: int,
    tp: int,
    rank: int,
    kind: MambaProjKind,
) -> list[tuple[int, int]]:
    """Rows of one rank's ``Shard(0)`` chunk inside the full projection.

    Shared groups store gate/x, ``1/tp`` of the B and C block, then dt.
    """
    if not _shares_groups(n_groups, tp):
        return _rank_row_ranges(
            num_heads, head_dim, n_groups, ssm_state_size, tp, rank, kind
        )
    intermediate = num_heads * head_dim
    local_inner = intermediate // tp
    local_heads = num_heads // tp
    group_rows = 2 * n_groups * ssm_state_size
    local_group_rows = group_rows // tp
    x0 = 0 if kind == "conv" else intermediate
    b0 = x0 + intermediate
    ranges = [] if kind == "conv" else [(rank * local_inner, (rank + 1) * local_inner)]
    ranges += [
        (x0 + rank * local_inner, x0 + (rank + 1) * local_inner),
        (b0 + rank * local_group_rows, b0 + (rank + 1) * local_group_rows),
    ]
    if kind == "in_proj":
        dt0 = b0 + group_rows
        ranges.append((dt0 + rank * local_heads, dt0 + (rank + 1) * local_heads))
    return ranges


def _row_order(
    num_heads: int,
    head_dim: int,
    n_groups: int,
    ssm_state_size: int,
    tp: int,
    kind: MambaProjKind,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for rank in range(tp):
        for start, stop in _stored_row_ranges(
            num_heads, head_dim, n_groups, ssm_state_size, tp, rank, kind
        ):
            parts.append(torch.arange(start, stop))
    return torch.cat(parts)


def _shard_marked_rows(
    module: nn.Module, name: str, row_order: torch.Tensor, device_mesh: DeviceMesh
) -> None:
    shard_reordered_rows(module, name, row_order, device_mesh)
    mark_mamba_tp_shard(getattr(module, name))


def _shard_marked_dim0(module: nn.Module, name: str, device_mesh: DeviceMesh) -> None:
    shard_dim0(module, name, device_mesh)
    mark_mamba_tp_shard(getattr(module, name))


def _in_proj_weight(mixer: Mamba2Mixer) -> nn.Parameter:
    return base_linear(mixer.in_proj).get_parameter("weight")


def _tp_full_width_norm(
    self: nn.Module, hidden_states: torch.Tensor, gate: torch.Tensor | None = None
) -> torch.Tensor:
    """``rmsnorm(x * silu(gate))`` over every head, with the variance summed over TP."""
    dtype = hidden_states.dtype
    values = hidden_states.to(torch.float32)
    if gate is not None:
        values = values * nn.functional.silu(gate.to(torch.float32))
    squares = SumOverTPRegion.apply(
        values.pow(2).sum(-1, keepdim=True), cast("dist.ProcessGroup", self._tp_group)
    )
    variance = squares / (values.shape[-1] * cast("int", self._tp_degree))
    values = values * torch.rsqrt(variance + cast("float", self.variance_epsilon))
    return cast("torch.Tensor", self.weight) * values.to(dtype)


def _pad_row_stride(
    _module: nn.Module, _args: tuple[object, ...], output: torch.Tensor
) -> torch.Tensor:
    # causal_conv1d's channel-last kernel reads x = in_proj[..., xBC] and needs
    # its row stride to be a multiple of 8.
    width = int(output.shape[-1])
    if width % 8 == 0:
        return output
    return nn.functional.pad(output, (0, -width % 8))[..., :width]


def _assemble_shared_groups(
    stored: torch.Tensor, mixer: Mamba2Mixer, name: str, kind: MambaProjKind
) -> torch.Tensor:
    """One rank's local mixer rows; B and C gather over TP, once when frozen.

    A trainable parameter's gather sums the B and C row grads over TP in backward.
    """
    num_heads, head_dim, n_groups, ssm_state_size = cast(
        "tuple[int, int, int, int]", mixer._tp_full_dims
    )
    tp = cast("int", mixer._tp_degree)
    rank = cast("DeviceMesh", mixer._tp_mesh).get_local_rank()
    local_inner = num_heads * head_dim // tp
    local_heads = num_heads // tp
    state = n_groups * ssm_state_size
    group0 = rank * local_heads // (num_heads // n_groups)
    if kind == "in_proj":
        gate, x, groups, dt = stored.split(
            [local_inner, local_inner, 2 * state // tp, local_heads]
        )
    else:
        x, groups = stored.split([local_inner, 2 * state // tp])
    full = gather_rows_once(
        mixer,
        name,
        groups,
        mixer.get_parameter(name).requires_grad,
        rank,
        cast("dist.ProcessGroup", mixer._tp_group),
    )
    b = full[group0 * ssm_state_size : (group0 + 1) * ssm_state_size]
    c = full[state + group0 * ssm_state_size : state + (group0 + 1) * ssm_state_size]
    if kind == "in_proj":
        return torch.cat([gate, x, b, c, dt])
    return torch.cat([x, b, c])


@contextmanager
def _local_mixer_sizes(mixer: Mamba2Mixer, tp: int) -> Iterator[None]:
    """Head, group, and conv sizes for one tensor-parallel forward."""
    num_heads, head_dim, n_groups, state_size = _mixer_dims(mixer)
    intermediate_size = mixer.intermediate_size
    conv_dim = mixer.conv_dim
    conv_in = mixer.conv1d.in_channels
    conv_out = mixer.conv1d.out_channels
    conv_groups = mixer.conv1d.groups
    local_heads = num_heads // tp
    local_groups = max(n_groups // tp, 1)
    local_intermediate = local_heads * head_dim
    local_conv = local_intermediate + 2 * local_groups * state_size
    # The mixer forward splits projections with these.
    mixer.num_heads = local_heads
    mixer.n_groups = local_groups
    mixer.intermediate_size = local_intermediate
    mixer.conv_dim = local_conv
    mixer.conv1d.in_channels = local_conv
    mixer.conv1d.out_channels = local_conv
    mixer.conv1d.groups = local_conv
    try:
        yield
    finally:
        mixer.num_heads = num_heads
        mixer.n_groups = n_groups
        mixer.intermediate_size = intermediate_size
        mixer.conv_dim = conv_dim
        mixer.conv1d.in_channels = conv_in
        mixer.conv1d.out_channels = conv_out
        mixer.conv1d.groups = conv_groups


def _call_local_forward(
    module: Mamba2Mixer,
    original: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> torch.Tensor:
    saved = module.forward
    was_training = module.training
    module.forward = original
    # The fused training scan normalizes only this rank's heads.
    module.training = was_training and not module._tp_norm_spans_ranks
    try:
        # Local matmul omits out_proj.bias. The full bias is added after the all-reduce.
        params = local_region_params(module, module.out_proj.bias)
        n_groups = cast("tuple[int, int, int, int]", module._tp_full_dims)[2]
        if _shares_groups(n_groups, cast("int", module._tp_degree)):
            for name, tensor in params.items():
                kind = _permuted_kind(name)
                if kind is not None:
                    params[name] = _assemble_shared_groups(tensor, module, name, kind)
        return functional_call(module, params, args, kwargs)
    finally:
        module.forward = saved
        module.training = was_training


def _install_mamba_tp_forward(mixer: Mamba2Mixer) -> None:
    if getattr(mixer, "_agilerl_mamba_tp", False):
        return
    # Context parallel wraps this and leaves ``forward`` as the tensor-parallel wrapper.
    object.__setattr__(mixer, "_mamba_inner_forward", mixer.forward)
    mixer.in_proj.register_forward_hook(_pad_row_stride)

    def tp_forward(self: Mamba2Mixer, *args: object, **kwargs: object) -> torch.Tensor:
        inner = cast("Callable[..., torch.Tensor]", self._mamba_inner_forward)
        tp = tp_degree(self)
        group = cast("dist.ProcessGroup", self._tp_group)
        args, kwargs = copy_input_to_region(args, kwargs, group)
        with _local_mixer_sizes(self, tp):
            output = _call_local_forward(self, inner, args, kwargs)
        return add_row_bias(ReduceFromTPRegion.apply(output, group), self.out_proj.bias)

    mixer.forward = MethodType(tp_forward, mixer)
    object.__setattr__(mixer, "_agilerl_mamba_tp", True)


def _require_mixer_split(mixer: Mamba2Mixer, tp: int, norm_group_size: int) -> bool:
    """Check ``mixer`` splits over ``tp``; return whether its gated norm spans ranks."""
    num_heads, _head_dim, n_groups, ssm_state_size = _mixer_dims(mixer)
    require_divisible(num_heads, tp, "mamba_num_heads")
    if _shares_groups(n_groups, tp):
        if tp % n_groups != 0:
            msg = f"n_groups ({n_groups}) and tp ({tp}) must divide one another"
            raise ValueError(msg)
        require_divisible(2 * n_groups * ssm_state_size, tp, "mamba B/C rows")
    intermediate = int(mixer.intermediate_size)
    if (intermediate // tp) % norm_group_size == 0:
        return False
    if norm_group_size != intermediate:
        msg = (
            f"gated norm groups of {norm_group_size} channels cross tensor-parallel "
            f"ranks (intermediate_size={intermediate}, tp={tp})"
        )
        raise ValueError(msg)
    return True


def shard_mamba_mixer(
    mixer: Mamba2Mixer, device_mesh: DeviceMesh, norm_group_size: int
) -> None:
    """Shard one Mamba2 mixer by head on ``device_mesh`` and all-reduce its output.

    :param mixer: Mamba2 mixer with ``in_proj``, ``conv1d``, ``dt_bias``,
        ``A_log``, ``D``, a gated ``norm`` and ``out_proj``.
    :type mixer: Mamba2Mixer
    :param device_mesh: Tensor-parallel mesh.
    :type device_mesh: DeviceMesh
    :param norm_group_size: Channels the gated norm averages over. A norm over
        every channel must compute ``rmsnorm(x * silu(gate))``.
    :type norm_group_size: int
    :return: None
    :rtype: None
    """
    tp = int(device_mesh.size())
    spans_ranks = _require_mixer_split(mixer, tp, norm_group_size)
    # The forward runs with local head and group counts set on the mixer.
    object.__setattr__(mixer, "_tp_full_dims", _mixer_dims(mixer))
    object.__setattr__(mixer, "_tp_norm_spans_ranks", spans_ranks)
    if spans_ranks:
        norm = cast("nn.Module", mixer.norm)
        stash_tp(norm, device_mesh)
        norm.forward = MethodType(_tp_full_width_norm, norm)
    weight = _in_proj_weight(mixer)
    if isinstance(weight, DTensor):
        stash_tp(mixer, device_mesh)
        _install_mamba_tp_forward(mixer)
        return
    num_heads, head_dim, n_groups, ssm_state_size = _mixer_dims(mixer)
    in_order = _row_order(num_heads, head_dim, n_groups, ssm_state_size, tp, "in_proj")
    conv_order = _row_order(num_heads, head_dim, n_groups, ssm_state_size, tp, "conv")
    in_linear = base_linear(mixer.in_proj)
    _shard_marked_rows(in_linear, "weight", in_order, device_mesh)
    if in_linear.bias is not None:
        _shard_marked_rows(in_linear, "bias", in_order, device_mesh)
    shard_lora_b_rows(mixer.in_proj, in_order, device_mesh)
    keep_lora_bank(mixer.in_proj, "lora_A", device_mesh)
    share_lora_dropout(mixer.in_proj, device_mesh)
    _shard_marked_rows(mixer.conv1d, "weight", conv_order, device_mesh)
    if mixer.conv1d.bias is not None:
        _shard_marked_rows(mixer.conv1d, "bias", conv_order, device_mesh)
    _shard_marked_dim0(mixer, "dt_bias", device_mesh)
    _shard_marked_dim0(mixer, "A_log", device_mesh)
    _shard_marked_dim0(mixer, "D", device_mesh)
    _shard_marked_dim0(mixer.norm, "weight", device_mesh)
    shard_row(mixer.out_proj, device_mesh)
    stash_tp(mixer, device_mesh)
    _install_mamba_tp_forward(mixer)


def _permuted_kind(relative: str) -> MambaProjKind | None:
    if relative in {
        "in_proj.weight",
        "in_proj.bias",
        "in_proj.base_layer.weight",
        "in_proj.base_layer.bias",
    } or is_lora_b_weight(relative, "in_proj"):
        return "in_proj"
    if relative in {"conv1d.weight", "conv1d.bias"}:
        return "conv"
    return None


def iter_permuted_mamba_params(
    model: nn.Module, is_mixer: IsMixer = is_nemotron_h_mixer
) -> list[tuple[str, Mamba2Mixer, MambaProjKind]]:
    """Tensor-parallel mixer params whose rows are grouped per rank.

    A sharded mixer stores these rows per rank, so its params are listed also
    while :func:`~agilerl.distributed.fsdp.gather_params` holds them dense.

    :param model: Module tree that may contain sharded Mamba2 mixers.
    :type model: nn.Module
    :param is_mixer: Which modules are this family's Mamba2 mixers.
    :type is_mixer: IsMixer
    :return: ``(fqn, mixer, kind)`` for ``in_proj``, its LoRA B, and ``conv1d``.
    :rtype: list[tuple[str, Mamba2Mixer, MambaProjKind]]
    """
    mixers = {
        name: cast("Mamba2Mixer", module)
        for name, module in model.named_modules()
        if is_mixer(module) and tp_degree(module) > 1
    }
    found: list[tuple[str, Mamba2Mixer, MambaProjKind]] = []
    for fqn, _ in model.named_parameters():
        owner: tuple[str, Mamba2Mixer] | None = None
        for name, mixer in mixers.items():
            if name and not (fqn == name or fqn.startswith(name + ".")):
                continue
            if owner is None or len(name) > len(owner[0]):
                owner = (name, mixer)
        if owner is None:
            continue
        mixer_name, mixer = owner
        relative = fqn[len(mixer_name) + 1 :] if mixer_name else fqn
        kind = _permuted_kind(relative)
        if kind is None:
            continue
        found.append((fqn, mixer, kind))
    return found


def mamba_row_orders(
    model: nn.Module, is_mixer: IsMixer = is_nemotron_h_mixer
) -> dict[str, torch.Tensor]:
    """Source row of each stored dim-0 row of the permuted mixer parameters.

    :param model: Model with tensor-parallel Mamba2 mixers.
    :type model: nn.Module
    :param is_mixer: Which modules are this family's Mamba2 mixers.
    :type is_mixer: IsMixer
    :return: Row order by live parameter FQN.
    :rtype: dict[str, torch.Tensor]
    """
    orders: dict[str, torch.Tensor] = {}
    for fqn, mixer, kind in iter_permuted_mamba_params(model, is_mixer):
        num_heads, head_dim, n_groups, ssm_state_size = _mixer_dims(mixer)
        orders[fqn] = _row_order(
            num_heads,
            head_dim,
            n_groups,
            ssm_state_size,
            cast("int", mixer._tp_degree),
            kind,
        )
    return orders


def realign_mamba_permuted_shards(
    model: nn.Module, is_mixer: IsMixer = is_nemotron_h_mixer
) -> None:
    """Copy per-rank gate, conv, and dt rows into tensor-parallel mixer shards.

    :param model: Model with tensor-parallel Mamba2 mixers.
    :type model: nn.Module
    :param is_mixer: Which modules are this family's Mamba2 mixers.
    :type is_mixer: IsMixer
    :return: None
    :rtype: None
    """
    with torch.no_grad():
        for fqn, order in sorted(mamba_row_orders(model, is_mixer).items()):
            permute_rows(cast("DTensor", model.get_parameter(fqn)), order)


def mark_mamba_tp_params(model: nn.Module) -> None:
    """Mark tensor-parallel mixer parameters; call again after they are replaced.

    :param model: Model that may contain sharded Mamba2 mixers.
    :type model: nn.Module
    :return: None
    :rtype: None
    """
    for module in model.modules():
        if not isinstance(module, NemotronHMamba2Mixer):
            continue
        mesh = getattr(module, "_tp_mesh", None)
        if mesh is None or tp_degree(module) <= 1:
            continue
        for param in module.parameters():
            if isinstance(param, DTensor) and on_tp_mesh(param):
                mark_mamba_tp_shard(param)


def apply_mamba_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh | None) -> int:
    """Shard NemotronH Mamba2 mixers on ``tp_mesh`` and all-reduce outputs.

    :param model: Module tree that may contain Mamba2 mixers.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh. ``None`` shards nothing.
    :type tp_mesh: DeviceMesh | None
    :return: How many mixers were sharded.
    :rtype: int
    """
    if tp_mesh is None or int(tp_mesh.size()) <= 1:
        return 0
    count = 0
    for module in model.modules():
        if not isinstance(module, NemotronHMamba2Mixer):
            continue
        # Shared B/C rows are not wired for the Nemotron-H norm and kernels.
        require_divisible(int(module.n_groups), int(tp_mesh.size()), "n_groups")
        shard_mamba_mixer(
            module, tp_mesh, int(module.intermediate_size) // int(module.n_groups)
        )
        count += 1
    return count


def _shard_attention(module: NemotronHAttention, mesh: DeviceMesh) -> None:
    config = module.config
    shard_attention_projections(
        module,
        mesh,
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(module.head_dim),
        module.q_proj,
        module.k_proj,
        module.v_proj,
        module.o_proj,
    )


def _shard_mlp(module: NemotronHMLP, mesh: DeviceMesh) -> None:
    require_divisible(
        int(module.intermediate_size), int(mesh.size()), "intermediate_size"
    )
    shard_column(module.up_proj, mesh)
    bias = shard_row(module.down_proj, mesh)
    stash_tp(module, mesh)
    install_reduce_forward(module, module.down_proj if bias is not None else None)


def _vision_head_count(module: nn.Module) -> int:
    for name in ("num_heads", "num_attention_heads"):
        value = module.__dict__.get(name)
        if isinstance(value, int):
            return value
    msg = f"{type(module).__name__} attention has no num_heads"
    raise ValueError(msg)


def _shard_vision_attention(module: nn.Module, mesh: DeviceMesh) -> None:
    num_q = _vision_head_count(module)
    num_kv_value = module.__dict__.get("num_key_value_heads", num_q)
    num_kv = int(num_kv_value)
    rows = int(base_weight(module.get_submodule("query")).shape[0])
    if rows % num_q != 0:
        msg = f"query rows ({rows}) must be divisible by num_heads ({num_q})"
        raise ValueError(msg)
    head_dim = module.__dict__.get("head_dim", rows // num_q)
    shard_attention_projections(
        module,
        mesh,
        num_q,
        num_kv,
        int(head_dim),
        module.get_submodule("query"),
        module.get_submodule("key"),
        module.get_submodule("value"),
        module.get_submodule("proj"),
    )


def _shard_vision_mlp(module: nn.Module, mesh: DeviceMesh) -> None:
    fc1 = module.get_submodule("fc1")
    fc2 = module.get_submodule("fc2")
    intermediate = int(base_weight(fc1).shape[0])
    require_divisible(intermediate, int(mesh.size()), "intermediate_size")
    shard_column(fc1, mesh)
    bias = shard_row(fc2, mesh)
    stash_tp(module, mesh)
    install_reduce_forward(module, fc2 if bias is not None else None)


def _named_linear(module: nn.Module, name: str) -> bool:
    child = getattr(module, name, None)
    return isinstance(child, nn.Module) and is_linear(child)


def _is_vision_attention(module: nn.Module) -> bool:
    return all(
        _named_linear(module, name) for name in ("query", "key", "value", "proj")
    )


def _is_vision_mlp(module: nn.Module) -> bool:
    return all(_named_linear(module, name) for name in ("fc1", "fc2"))


def _has_latent_projections(module: nn.Module) -> bool:
    return isinstance(
        getattr(module, "fc1_latent_proj", None), nn.Module
    ) and isinstance(getattr(module, "fc2_latent_proj", None), nn.Module)


def _shard_latent(module: nn.Module, mesh: DeviceMesh) -> bool:
    fc1 = module.get_submodule("fc1_latent_proj")
    fc2 = module.get_submodule("fc2_latent_proj")
    if not is_linear(fc1) or not is_linear(fc2):
        return False
    latent = int(base_weight(fc1).shape[0])
    require_divisible(latent, int(mesh.size()), "moe_latent_size")
    shard_column(fc1, mesh)
    stash_tp(fc1, mesh)
    install_gather_forward(fc1)
    bias = shard_row(fc2, mesh)
    stash_tp(fc2, mesh)
    install_split_forward(fc2, fc2 if bias is not None else None)
    return True


def apply_nemotron_h_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh) -> int:
    """Shard Nemotron-H mixers, attention, MLPs, latent MoE projections, and vision tower.

    :param model: Module tree that may contain Nemotron-H blocks.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh.
    :type tp_mesh: DeviceMesh
    :return: How many modules were sharded.
    :rtype: int
    """
    count = apply_mamba_tensor_parallel(model, tp_mesh)
    for name, module in model.named_modules():
        if isinstance(module, NemotronHAttention):
            _shard_attention(module, tp_mesh)
        elif isinstance(module, NemotronHMLP):
            _shard_mlp(module, tp_mesh)
            # The MoE block ends on the shared expert's reduce plus an add, so
            # its recompute stops before that reduce.
            if name.rsplit(".", 1)[-1] == "shared_experts":
                keep_collectives_eager(module)
        elif _has_latent_projections(module):
            if not _shard_latent(module, tp_mesh):
                continue
        elif "vision_model" in name and _is_vision_attention(module):
            _shard_vision_attention(module, tp_mesh)
        elif "vision_model" in name and _is_vision_mlp(module):
            _shard_vision_mlp(module, tp_mesh)
        else:
            continue
        count += 1
    return count


def restore_nemotron_h_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Regroup mixer rows per rank and re-mark mixer shards after weights load.

    :param model: Model with tensor-parallel Mamba2 mixers.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh.
    :type tp_mesh: DeviceMesh
    :return: None
    :rtype: None
    """
    realign_mamba_permuted_shards(model)
    apply_mamba_tensor_parallel(model, tp_mesh)
    mark_mamba_tp_params(model)


NEMOTRON_H_TENSOR_PARALLEL_PLAN = TensorParallelPlan(
    shard=apply_nemotron_h_tensor_parallel,
    restore=restore_nemotron_h_tensor_parallel,
    row_orders=mamba_row_orders,
)
