# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel Mamba2 mixers.

``in_proj`` runs on the sequence shard, an all-to-all turns the projected
states into ``[B, L, P/cp]``, conv and SSM run on that head shard of the full
sequence, and an all-to-all brings the output back to the sequence shard for
``out_proj``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from agilerl.architectures.nemotron_h.mamba import MEM_EFF_ATTR
from agilerl.distributed.context_parallel.mixer import (
    LocalParamSlot,
    assign_view,
    call_local,
    drop_sequence_shard_mask,
    local_param_slots,
    local_tensor,
)
from agilerl.distributed.context_parallel.sequence import (
    all_to_all_hidden_to_seq,
    all_to_all_seq_to_hidden,
)
from agilerl.distributed.context_parallel.ulysses import ULYSSES_PARAMS
from agilerl.distributed.tensor_parallel import require_divisible, tp_degree

if TYPE_CHECKING:
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
        GraniteMoeHybridMambaLayer,
    )
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        NemotronHMamba2Mixer,
    )

    Mamba2Mixer = NemotronHMamba2Mixer | GraniteMoeHybridMambaLayer
MAMBA_CP_SAVED_ATTR = "_mamba_cp_saved"


def _require_groups_split(n_groups: int, cp: int) -> None:
    if n_groups % cp != 0 and cp % n_groups != 0:
        msg = f"n_groups ({n_groups}) and cp ({cp}) must divide one another"
        raise ValueError(msg)


def _rank_groups(num_heads: int, n_groups: int, cp: int, rank: int) -> tuple[int, int]:
    """First SSM group and group count for ``rank``'s heads.

    With fewer groups than ``cp``, every rank whose heads share a group holds it.
    """
    first = rank * (num_heads // cp) // (num_heads // n_groups)
    return first, max(n_groups // cp, 1)


@dataclass(frozen=True)
class MambaDims:
    """Current head and SSM state sizes of a Mamba2 mixer."""

    num_heads: int
    head_dim: int
    n_groups: int
    ssm_state_size: int


def _mamba_dims(mixer: Mamba2Mixer) -> MambaDims:
    # HF types these optional; a built Mamba2 mixer always sets them.
    return MambaDims(
        cast("int", mixer.num_heads),
        cast("int", mixer.head_dim),
        cast("int", mixer.n_groups),
        cast("int", mixer.ssm_state_size),
    )


@dataclass(frozen=True)
class MambaFullState:
    """Full-width mixer weights and sizes swapped out during a context-parallel scan."""

    in_proj: nn.Module
    out_proj: nn.Module
    norm_module: nn.Module
    conv_weight: torch.Tensor
    conv_bias: torch.Tensor | None
    dt_bias: torch.Tensor
    a_log: torch.Tensor
    d: torch.Tensor
    norm_weight: torch.Tensor
    num_heads: int
    n_groups: int
    intermediate_size: int
    conv_dim: int
    conv_in: int
    conv_out: int
    conv_groups: int
    use_mem_eff_path: bool | None


def _save_full_state(mixer: Mamba2Mixer) -> None:
    dims = _mamba_dims(mixer)
    state = MambaFullState(
        in_proj=mixer.in_proj,
        out_proj=mixer.out_proj,
        norm_module=mixer.norm,
        conv_weight=mixer.conv1d.weight,
        conv_bias=mixer.conv1d.bias,
        dt_bias=mixer.dt_bias,
        a_log=mixer.A_log,
        d=mixer.D,
        norm_weight=mixer.norm.weight,
        num_heads=dims.num_heads,
        n_groups=dims.n_groups,
        intermediate_size=mixer.intermediate_size,
        conv_dim=mixer.conv_dim,
        conv_in=mixer.conv1d.in_channels,
        conv_out=mixer.conv1d.out_channels,
        conv_groups=mixer.conv1d.groups,
        use_mem_eff_path=getattr(mixer, MEM_EFF_ATTR, None),
    )
    object.__setattr__(mixer, MAMBA_CP_SAVED_ATTR, state)


def _restore_full_state(mixer: Mamba2Mixer) -> None:
    state: MambaFullState | None = getattr(mixer, MAMBA_CP_SAVED_ATTR, None)
    if state is None:
        return
    nn.Module.__setattr__(mixer, "in_proj", state.in_proj)
    nn.Module.__setattr__(mixer, "out_proj", state.out_proj)
    nn.Module.__setattr__(mixer, "norm", state.norm_module)
    assign_view(mixer.conv1d, "weight", state.conv_weight)
    assign_view(mixer.conv1d, "bias", state.conv_bias)
    assign_view(mixer, "dt_bias", state.dt_bias)
    assign_view(mixer, "A_log", state.a_log)
    assign_view(mixer, "D", state.d)
    assign_view(mixer.norm, "weight", state.norm_weight)
    mixer.num_heads = state.num_heads
    mixer.n_groups = state.n_groups
    mixer.intermediate_size = state.intermediate_size
    mixer.conv_dim = state.conv_dim
    mixer.conv1d.in_channels = state.conv_in
    mixer.conv1d.out_channels = state.conv_out
    mixer.conv1d.groups = state.conv_groups
    if state.use_mem_eff_path is not None:
        object.__setattr__(mixer, MEM_EFF_ATTR, state.use_mem_eff_path)
    object.__setattr__(mixer, MAMBA_CP_SAVED_ATTR, None)


def _set_mixer_sizes(
    mixer: Mamba2Mixer,
    num_heads: int,
    n_groups: int,
    intermediate_size: int,
    conv_dim: int,
) -> None:
    mixer.num_heads = num_heads
    mixer.n_groups = n_groups
    mixer.intermediate_size = intermediate_size
    mixer.conv_dim = conv_dim


def _install_local_shard(
    mixer: Mamba2Mixer,
    slices: dict[str, torch.Tensor],
    local_heads: int,
    local_groups: int,
    local_inner: int,
    local_conv: int,
) -> None:
    """Point the conv, SSM params and sizes at this rank's head shard."""
    assign_view(mixer.conv1d, "weight", slices["conv_weight"])
    if "conv_bias" in slices:
        assign_view(mixer.conv1d, "bias", slices["conv_bias"])
    assign_view(mixer, "dt_bias", slices["dt_bias"])
    assign_view(mixer, "A_log", slices["A_log"])
    assign_view(mixer, "D", slices["D"])
    _set_mixer_sizes(mixer, local_heads, local_groups, local_inner, local_conv)
    mixer.conv1d.in_channels = local_conv
    mixer.conv1d.out_channels = local_conv
    mixer.conv1d.groups = local_conv


class MixerInProjBypass(nn.Module):
    """Stand-in for ``in_proj`` that returns an already-projected tensor."""

    value: torch.Tensor

    def __init__(self, value: torch.Tensor, like: nn.Module) -> None:
        super().__init__()
        # Activations and the borrowed Linear weight stay off this module tree.
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "weight", getattr(like, "weight", None))
        object.__setattr__(self, "bias", getattr(like, "bias", None))

    def forward(self, _hidden: torch.Tensor) -> torch.Tensor:
        return self.value


class MixerOutProjThenSeq(nn.Module):
    """``out_proj`` after all-to-all back to the sequence shard."""

    linear: nn.Module
    slots: tuple[LocalParamSlot, ...]
    cp: int
    group: dist.ProcessGroup

    def __init__(
        self,
        out_proj: nn.Module,
        slots: tuple[LocalParamSlot, ...],
        cp: int,
        group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "linear", out_proj)
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "cp", cp)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "weight", getattr(out_proj, "weight", None))
        object.__setattr__(self, "bias", getattr(out_proj, "bias", None))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        seq = all_to_all_hidden_to_seq(value, self.cp, self.group)
        return call_local(self.linear, self.slots, seq)


class MixerGatedNormThenSeq(nn.Module):
    """Gated norm whose groups span ranks: all-to-all the head shard, then norm.

    An fp32 ``hidden_states`` is gated before the all-to-all, so
    ``norm(x, gate)`` must equal ``norm(x * silu(gate))``, as gate-before-norm
    RMSNorms do.
    """

    norm: nn.Module
    slots: tuple[LocalParamSlot, ...]
    cp: int
    group: dist.ProcessGroup

    def __init__(
        self,
        norm: nn.Module,
        slots: tuple[LocalParamSlot, ...],
        cp: int,
        group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "norm", norm)
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "cp", cp)
        object.__setattr__(self, "group", group)

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if hidden_states.dtype == torch.float32:
            gated = hidden_states * nn.functional.silu(gate.float())
            seq = all_to_all_hidden_to_seq(gated, self.cp, self.group)
            return call_local(self.norm, self.slots, seq)
        # The norm forms the product in fp32 and rounds its output to the input
        # dtype, so a narrower ``hidden_states`` crosses with its gate.
        both = torch.cat([hidden_states, gate])
        seq = all_to_all_hidden_to_seq(both, self.cp, self.group)
        hidden_seq, gate_seq = seq.chunk(2)
        return call_local(self.norm, self.slots, hidden_seq, gate_seq)


@dataclass(frozen=True)
class MambaCpIndex:
    """Device index tensors of one rank's context-parallel Mamba scan.

    :param groups: SSM groups of every rank's ``B`` and ``C``, in rank order.
    :param conv: This rank's conv channels of ``[x | B | C]``.
    """

    groups: torch.Tensor
    conv: torch.Tensor


def mamba_cp_group_index(
    num_heads: int, n_groups: int, cp: int, device: torch.device | str
) -> torch.Tensor:
    """SSM groups of every rank's ``B`` and ``C``, in rank order."""
    return torch.cat(
        [
            torch.arange(first, first + count, device=device)
            for first, count in (
                _rank_groups(num_heads, n_groups, cp, rank) for rank in range(cp)
            )
        ]
    )


def mamba_cp_index(
    dims: MambaDims, cp: int, rank: int, device: torch.device | str
) -> MambaCpIndex:
    """Index tensors ``rank`` selects its head shard with, built once on ``device``.

    :param dims: Heads and SSM groups the CP group splits.
    :param cp: Context-parallel degree.
    :param rank: This rank in the CP group.
    :param device: Device the scan runs on.
    :return: Group and conv-channel indices.
    """
    d_inner = dims.num_heads * dims.head_dim
    group_state = dims.n_groups * dims.ssm_state_size
    local_inner = d_inner // cp
    group0, local_groups = _rank_groups(dims.num_heads, dims.n_groups, cp, rank)
    b_start = d_inner + group0 * dims.ssm_state_size
    local_state = local_groups * dims.ssm_state_size
    conv = torch.cat(
        [
            torch.arange(rank * local_inner, (rank + 1) * local_inner, device=device),
            torch.arange(b_start, b_start + local_state, device=device),
            torch.arange(
                b_start + group_state,
                b_start + group_state + local_state,
                device=device,
            ),
        ]
    )
    return MambaCpIndex(
        mamba_cp_group_index(dims.num_heads, dims.n_groups, cp, device), conv
    )


def pack_mamba_projected_for_cp(
    projected: torch.Tensor,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    ssm_state_size: int,
    cp: int,
    group_index: torch.Tensor,
) -> torch.Tensor:
    """Reorder ``in_proj`` so a hidden-dim all-to-all yields one head shard.

    ``in_proj`` is ``[gate | x | B | C | dt]``. Splitting that last dim into
    ``cp`` contiguous chunks cuts across those blocks, so each rank's SSM
    would mix another rank's gate with its own dt. Pack as ``cp`` concatenated
    local projections ``[gate_r | x_r | B_r | C_r | dt_r]``; then
    :func:`all_to_all_seq_to_hidden` matches the layout ``torch_forward``
    splits after shrinking ``num_heads`` / ``n_groups``. With fewer groups
    than ``cp``, each rank's ``B_r`` / ``C_r`` is the group its heads share.
    ``group_index`` is :func:`mamba_cp_group_index` of these sizes.
    """
    if cp <= 1:
        return projected
    batch, seq, width = projected.shape
    d_inner = num_heads * head_dim
    group_state = n_groups * ssm_state_size
    conv_dim = d_inner + 2 * group_state
    expected = d_inner + conv_dim + num_heads
    if width != expected:
        msg = (
            f"mamba in_proj last dim ({width}) != gate+xBC+dt ({expected}) "
            f"for heads={num_heads} groups={n_groups} cp={cp}"
        )
        raise ValueError(msg)
    local_inner = d_inner // cp
    local_heads = num_heads // cp
    gate, xbc, dt = projected.split([d_inner, conv_dim, num_heads], dim=-1)
    x_proj, b_proj, c_proj = xbc.split([d_inner, group_state, group_state], dim=-1)
    b_ranks = b_proj.reshape(batch, seq, n_groups, ssm_state_size).index_select(
        2, group_index
    )
    c_ranks = c_proj.reshape(batch, seq, n_groups, ssm_state_size).index_select(
        2, group_index
    )
    packed = torch.cat(
        [
            gate.reshape(batch, seq, cp, local_inner),
            x_proj.reshape(batch, seq, cp, local_inner),
            b_ranks.reshape(batch, seq, cp, -1),
            c_ranks.reshape(batch, seq, cp, -1),
            dt.reshape(batch, seq, cp, local_heads),
        ],
        dim=-1,
    )
    return packed.reshape(batch, seq, -1)


def _slice_mamba_ssm(
    mixer: Mamba2Mixer, cp: int, rank: int, conv_index: torch.Tensor
) -> dict[str, torch.Tensor]:
    dims = _mamba_dims(mixer)
    local_heads = dims.num_heads // cp
    local_inner = local_heads * dims.head_dim
    head0 = rank * local_heads
    slices: dict[str, torch.Tensor] = {
        "conv_weight": local_tensor(mixer.conv1d.weight).index_select(0, conv_index),
        "dt_bias": local_tensor(mixer.dt_bias)[head0 : head0 + local_heads],
        "A_log": local_tensor(mixer.A_log)[head0 : head0 + local_heads],
        "D": local_tensor(mixer.D)[head0 : head0 + local_heads],
        "norm": local_tensor(mixer.norm.weight)[
            rank * local_inner : rank * local_inner + local_inner
        ],
    }
    if mixer.conv1d.bias is not None:
        slices["conv_bias"] = local_tensor(mixer.conv1d.bias).index_select(
            0, conv_index
        )
    return slices


def _call_without_fused_scan(
    module: nn.Module,
    scan: Callable[..., torch.Tensor],
    hidden: torch.Tensor,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> torch.Tensor:
    """Call the eager scan. The fused training return applies ``out_proj`` itself."""
    was_training = module.training
    module.training = False
    try:
        return scan(hidden, *args, **kwargs)
    finally:
        module.training = was_training


def _pad_row_stride_to_8(projected: torch.Tensor) -> torch.Tensor:
    # causal_conv1d's channel-last kernel reads x = in_proj[..., xBC] and needs
    # its row stride to be a multiple of 8.
    width = projected.shape[-1]
    if width % 8 == 0:
        return projected
    return nn.functional.pad(projected, (0, -width % 8))[..., :width]


def _cp_mamba_scan(
    mixer: Mamba2Mixer,
    scan: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    projected: torch.Tensor,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    cp: int,
    group: dist.ProcessGroup,
    heads: int,
    groups: int,
    local_norm: bool,
    seq_slots: tuple[LocalParamSlot, ...],
    index: MambaCpIndex,
) -> torch.Tensor:
    """Run ``scan`` on this rank's head shard of the full sequence, then restore ``mixer``.

    :param mixer: Mamba2 mixer whose full-width state is swapped out for the call.
    :param scan: Mixer forward that runs conv, SSM, norm, and ``out_proj``.
    :param hidden_states: Sequence-shard hidden states.
    :param projected: ``in_proj`` of ``hidden_states``.
    :param args: Positional mixer arguments after the hidden states.
    :param kwargs: Keyword mixer arguments.
    :param cp: Context-parallel degree.
    :param group: Context-parallel process group.
    :param heads: Mamba heads this CP group splits.
    :param groups: SSM groups this CP group splits.
    :param local_norm: Whether each rank holds whole gated-norm groups.
    :param seq_slots: :func:`local_param_slots` of ``out_proj`` when
        ``local_norm``, else of ``norm``: the module run on the sequence shard.
    :param index: :func:`mamba_cp_index` of ``heads`` and ``groups``.
    :return: Mixer output on the sequence shard.
    """
    dims = _mamba_dims(mixer)
    head_dim = dims.head_dim
    ssm_state = dims.ssm_state_size
    rank = dist.get_rank(group)
    local_heads = heads // cp
    local_groups = max(groups // cp, 1)
    local_inner = local_heads * head_dim
    local_conv = local_inner + 2 * local_groups * ssm_state
    in_proj = mixer.in_proj
    out_proj = mixer.out_proj
    packed = pack_mamba_projected_for_cp(
        projected,
        num_heads=heads,
        head_dim=head_dim,
        n_groups=groups,
        ssm_state_size=ssm_state,
        cp=cp,
        group_index=index.groups,
    )
    projected_hp = _pad_row_stride_to_8(all_to_all_seq_to_hidden(packed, cp, group))
    _save_full_state(mixer)
    inner = heads * head_dim
    _set_mixer_sizes(mixer, heads, groups, inner, inner + 2 * groups * ssm_state)
    slices = _slice_mamba_ssm(mixer, cp, rank, index.conv)
    # Register stand-ins in ``_modules`` so cuda_kernels_forward / torch_forward
    # call them. The fused scan reads ``out_proj.weight`` and skips that call.
    nn.Module.__setattr__(mixer, "in_proj", MixerInProjBypass(projected_hp, in_proj))
    if local_norm:
        nn.Module.__setattr__(
            mixer, "out_proj", MixerOutProjThenSeq(out_proj, seq_slots, cp, group)
        )
        assign_view(mixer.norm, "weight", slices["norm"])
    else:
        nn.Module.__setattr__(
            mixer, "norm", MixerGatedNormThenSeq(mixer.norm, seq_slots, cp, group)
        )
    _install_local_shard(
        mixer, slices, local_heads, local_groups, local_inner, local_conv
    )
    object.__setattr__(mixer, MEM_EFF_ATTR, False)
    # The scan reads only the shape and dtype; in_proj is bypassed.
    dummy = hidden_states.new_zeros(()).expand(
        hidden_states.shape[0],
        projected_hp.shape[1],
        hidden_states.shape[-1],
    )
    scan_args, scan_kwargs = drop_sequence_shard_mask(args, kwargs)
    try:
        return _call_without_fused_scan(mixer, scan, dummy, scan_args, scan_kwargs)
    finally:
        _restore_full_state(mixer)


def _install_composed_mamba_cp(
    mixer: Mamba2Mixer,
    cp_mesh: DeviceMesh,
    local_norm: bool,
) -> None:
    inner = cast("Callable[..., torch.Tensor]", mixer._mamba_inner_forward)
    object.__setattr__(mixer, MEM_EFF_ATTR, False)
    tp = tp_degree(mixer)
    dims = _mamba_dims(mixer)
    tp_heads = dims.num_heads // tp
    tp_groups = max(dims.n_groups // tp, 1)
    in_proj_slots = local_param_slots(mixer.in_proj)
    seq_slots = local_param_slots(mixer.out_proj if local_norm else mixer.norm)
    cp = int(cp_mesh.size())
    group = cp_mesh.get_group()
    index = mamba_cp_index(
        MambaDims(tp_heads, dims.head_dim, tp_groups, dims.ssm_state_size),
        cp,
        int(cp_mesh.get_local_rank()),
        cp_mesh.device_type,
    )

    def cp_forward(
        self: Mamba2Mixer, hidden_states: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        # Training publishes cu_seqlens; generate and other full-sequence paths do not.
        if "cu_seqlens" not in ULYSSES_PARAMS:
            return inner(hidden_states, *args, **kwargs)
        _restore_full_state(self)
        projected = call_local(self.in_proj, in_proj_slots, hidden_states)
        return _cp_mamba_scan(
            self,
            inner,
            hidden_states,
            projected,
            args,
            kwargs,
            cp,
            group,
            tp_heads,
            tp_groups,
            local_norm,
            seq_slots,
            index,
        )

    object.__setattr__(mixer, "_mamba_inner_forward", MethodType(cp_forward, mixer))
    object.__setattr__(mixer, "_agilerl_mamba_cp", True)


def install_mamba_context_parallel(
    mixer: Mamba2Mixer, cp_mesh: DeviceMesh, norm_group_size: int | None = None
) -> bool:
    """Run a Mamba2 mixer's conv and scan on a head shard of the full sequence.

    :param mixer: Mamba2 mixer with ``in_proj``, ``conv1d``, ``dt_bias``,
        ``A_log``, ``D``, a gated ``norm`` and ``out_proj``.
    :param cp_mesh: Context-parallel mesh. Size is the CP degree.
    :param norm_group_size: Channels the gated norm averages over. ``None``
        means one SSM group's heads.
    :return: True when this call wrapped ``mixer``.
    """
    if getattr(mixer, "_agilerl_mamba_cp", False):
        return False
    cp = int(cp_mesh.size())
    group = cp_mesh.get_group()
    dims = _mamba_dims(mixer)
    if norm_group_size is None:
        norm_group_size = int(mixer.intermediate_size) // dims.n_groups
    if getattr(mixer, "_agilerl_mamba_tp", False):
        tp = tp_degree(mixer)
        require_divisible(dims.num_heads, tp * cp, "mamba_num_heads", "tp * cp")
        _require_groups_split(max(dims.n_groups // tp, 1), cp)
        local_norm = (int(mixer.intermediate_size) // (tp * cp)) % norm_group_size == 0
        _install_composed_mamba_cp(mixer, cp_mesh, local_norm)
        return True
    require_divisible(dims.num_heads, cp, "mamba_num_heads", "cp")
    _require_groups_split(dims.n_groups, cp)
    original = mixer.forward
    local_inner = dims.num_heads // cp * dims.head_dim
    # A rank holding whole norm groups normalizes its heads; otherwise the norm
    # runs on the sequence shard after the all-to-all back.
    local_norm = local_inner % norm_group_size == 0
    seq_slots = local_param_slots(mixer.out_proj if local_norm else mixer.norm)
    index = mamba_cp_index(dims, cp, int(cp_mesh.get_local_rank()), cp_mesh.device_type)

    def cp_forward(
        self: Mamba2Mixer, hidden_states: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        # Training publishes cu_seqlens; generate and other full-sequence paths do not.
        if "cu_seqlens" not in ULYSSES_PARAMS:
            return original(hidden_states, *args, **kwargs)
        _restore_full_state(self)
        full = _mamba_dims(self)
        projected = self.in_proj(hidden_states)
        self.forward = original
        try:
            return _cp_mamba_scan(
                self,
                original,
                hidden_states,
                projected,
                args,
                kwargs,
                cp,
                group,
                full.num_heads,
                full.n_groups,
                local_norm,
                seq_slots,
                index,
            )
        finally:
            self.forward = MethodType(cp_forward, self)

    mixer.forward = MethodType(cp_forward, mixer)
    object.__setattr__(mixer, "_agilerl_mamba_cp", True)
    return True
