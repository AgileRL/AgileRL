# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel mixer wrap primitives.

Family catalog plans call :func:`install_mamba_context_parallel` or
:func:`install_gdn_context_parallel` on their own mixer classes.
:func:`apply_mixer_context_parallel` runs every catalog plan.

Mamba: in_proj on the sequence shard, all-to-all projected states to
``[B, L, P/cp]``, local conv/SSM, all-to-all back, out_proj on the shard.
GDN: gather the residual to full sequence, local mixer, shard back.
"""

from __future__ import annotations

from types import MethodType

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.func import functional_call

from agilerl.architectures import family_context_parallel_plans
from agilerl.distributed.context_parallel.sequence import (
    all_to_all_hidden_to_seq,
    all_to_all_seq_to_hidden,
    gather_for_cp,
    shard_for_cp,
    validate_cp_degree,
)


def _require_divisible(value: int, divisor: int, name: str) -> None:
    if value % divisor != 0:
        msg = f"{name} ({value}) must be divisible by {divisor}"
        raise ValueError(msg)


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    if isinstance(value, DTensor):
        return value.to_local()
    return value


def _call_local(
    module: nn.Module, *args: object, **kwargs: object
) -> torch.Tensor:
    params = dict(module.named_parameters())
    row_ranges = {
        name: child.tp_row_ranges
        for name, child in module.named_modules()
        if hasattr(child, "tp_row_ranges")
    }
    has_dtensor = any(isinstance(param, DTensor) for param in params.values())
    local_params: dict[str, torch.Tensor] = {}
    slice_rows = False
    for name, param in params.items():
        tensor = _local_tensor(param)
        ranges = row_ranges.get(name.rpartition(".")[0])
        if ranges is not None:
            local_rows = sum(stop - start for start, stop in ranges)
            # LoRA B is replicated full-width; mixer CP recompute calls in_proj
            # outside the TP region, so those rows still need this rank's slice.
            if tensor.shape[0] != local_rows:
                tensor = torch.cat([tensor[start:stop] for start, stop in ranges])
                slice_rows = True
        local_params[name] = tensor
    if not has_dtensor and not slice_rows:
        return module(*args, **kwargs)
    return functional_call(module, local_params, args, kwargs)


def _assign_view(module: nn.Module, name: str, value: torch.Tensor) -> None:
    """Put a view in a Parameter slot without wrapping it (keeps autograd)."""
    if name in module._parameters:
        module._parameters[name] = value
        return
    setattr(module, name, value)


class MixerInProjBypass(nn.Module):
    """Stand-in for ``in_proj`` that returns an already-projected tensor."""

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

    def __init__(self, out_proj: nn.Module, cp: int, group: dist.ProcessGroup) -> None:
        super().__init__()
        object.__setattr__(self, "linear", out_proj)
        object.__setattr__(self, "cp", cp)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "weight", getattr(out_proj, "weight", None))
        object.__setattr__(self, "bias", getattr(out_proj, "bias", None))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        seq = all_to_all_hidden_to_seq(value, self.cp, self.group)
        return _call_local(self.linear, seq)


def _slice_mamba_ssm(mixer: nn.Module, cp: int, rank: int) -> dict[str, torch.Tensor]:
    num_heads = int(mixer.num_heads)
    head_dim = int(mixer.head_dim)
    n_groups = int(mixer.n_groups)
    d_state = int(mixer.ssm_state_size)
    d_inner = num_heads * head_dim
    local_heads = num_heads // cp
    local_inner = local_heads * head_dim
    local_groups = n_groups // cp
    head0 = rank * local_heads
    group0 = rank * local_groups
    conv_weight = _local_tensor(mixer.conv1d.weight)
    conv_index = torch.cat(
        [
            torch.arange(rank * local_inner, rank * local_inner + local_inner),
            torch.arange(
                d_inner + group0 * d_state,
                d_inner + group0 * d_state + local_groups * d_state,
            ),
            torch.arange(
                d_inner + n_groups * d_state + group0 * d_state,
                d_inner
                + n_groups * d_state
                + group0 * d_state
                + local_groups * d_state,
            ),
        ]
    ).to(device=conv_weight.device)
    slices: dict[str, torch.Tensor] = {
        "conv_weight": conv_weight.index_select(0, conv_index),
        "dt_bias": _local_tensor(mixer.dt_bias)[head0 : head0 + local_heads],
        "A_log": _local_tensor(mixer.A_log)[head0 : head0 + local_heads],
        "D": _local_tensor(mixer.D)[head0 : head0 + local_heads],
        "norm": _local_tensor(mixer.norm.weight)[
            rank * local_inner : rank * local_inner + local_inner
        ],
    }
    if mixer.conv1d.bias is not None:
        slices["conv_bias"] = _local_tensor(mixer.conv1d.bias).index_select(
            0, conv_index
        )
    return slices


def _install_composed_mamba_cp(
    mixer: nn.Module, cp: int, group: dist.ProcessGroup
) -> None:
    original_torch = mixer.torch_forward
    mixer.use_mem_eff_path = False
    tp = int(getattr(mixer, "_tp_degree", 1) or 1)
    tp_heads = int(mixer.num_heads) // tp
    tp_groups = int(mixer.n_groups) // tp
    head_dim = int(mixer.head_dim)
    ssm_state = int(mixer.ssm_state_size)

    def restore_full(self: nn.Module) -> None:
        saved = getattr(self, "_mamba_cp_saved", None)
        if saved is None:
            return
        nn.Module.__setattr__(self, "in_proj", saved["in_proj"])
        nn.Module.__setattr__(self, "out_proj", saved["out_proj"])
        _assign_view(self.conv1d, "weight", saved["conv_weight"])
        if saved["conv_bias"] is not None:
            _assign_view(self.conv1d, "bias", saved["conv_bias"])
        else:
            self.conv1d.bias = None
        _assign_view(self, "dt_bias", saved["dt_bias"])
        _assign_view(self, "A_log", saved["A_log"])
        _assign_view(self, "D", saved["D"])
        _assign_view(self.norm, "weight", saved["norm"])
        self.num_heads = saved["num_heads"]
        self.n_groups = saved["n_groups"]
        self.intermediate_size = saved["intermediate_size"]
        self.conv_dim = saved["conv_dim"]
        self.conv1d.in_channels = saved["conv_in"]
        self.conv1d.out_channels = saved["conv_out"]
        self.conv1d.groups = saved["conv_groups"]
        object.__setattr__(self, "_mamba_cp_saved", None)

    def cp_torch_forward(
        self: nn.Module, hidden_states: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        restore_full(self)
        rank = dist.get_rank(group)
        local_heads = tp_heads // cp
        local_groups = tp_groups // cp
        local_inner = local_heads * head_dim
        local_conv = local_inner + 2 * local_groups * ssm_state
        in_proj = self.in_proj
        out_proj = self.out_proj
        projected = _call_local(in_proj, hidden_states)
        projected_hp = all_to_all_seq_to_hidden(projected, cp, group)
        saved = {
            "in_proj": in_proj,
            "out_proj": out_proj,
            "conv_weight": self.conv1d.weight,
            "conv_bias": self.conv1d.bias,
            "dt_bias": self.dt_bias,
            "A_log": self.A_log,
            "D": self.D,
            "norm": self.norm.weight,
            "num_heads": self.num_heads,
            "n_groups": self.n_groups,
            "intermediate_size": self.intermediate_size,
            "conv_dim": self.conv_dim,
            "conv_in": self.conv1d.in_channels,
            "conv_out": self.conv1d.out_channels,
            "conv_groups": self.conv1d.groups,
        }
        self.num_heads = tp_heads
        self.n_groups = tp_groups
        self.intermediate_size = tp_heads * head_dim
        self.conv_dim = self.intermediate_size + 2 * tp_groups * ssm_state
        slices = _slice_mamba_ssm(self, cp, rank)
        nn.Module.__setattr__(self, "in_proj", MixerInProjBypass(projected_hp, in_proj))
        nn.Module.__setattr__(self, "out_proj", MixerOutProjThenSeq(out_proj, cp, group))
        _assign_view(self.conv1d, "weight", slices["conv_weight"])
        if "conv_bias" in slices:
            _assign_view(self.conv1d, "bias", slices["conv_bias"])
        _assign_view(self, "dt_bias", slices["dt_bias"])
        _assign_view(self, "A_log", slices["A_log"])
        _assign_view(self, "D", slices["D"])
        _assign_view(self.norm, "weight", slices["norm"])
        self.num_heads = local_heads
        self.n_groups = local_groups
        self.intermediate_size = local_inner
        self.conv_dim = local_conv
        self.conv1d.in_channels = local_conv
        self.conv1d.out_channels = local_conv
        self.conv1d.groups = local_conv
        object.__setattr__(self, "_mamba_cp_saved", saved)
        dummy = hidden_states.new_zeros(
            hidden_states.shape[0],
            projected_hp.shape[1],
            hidden_states.shape[-1],
        )
        try:
            return original_torch(dummy, *args, **kwargs)
        finally:
            restore_full(self)

    mixer.torch_forward = MethodType(cp_torch_forward, mixer)
    # HF forward calls cuda_kernels_forward on CUDA when the fast path is loaded.
    if hasattr(mixer, "cuda_kernels_forward"):
        mixer.cuda_kernels_forward = mixer.torch_forward
    object.__setattr__(mixer, "_agilerl_mamba_cp", True)


def install_mamba_context_parallel(mixer: nn.Module, cp_mesh: DeviceMesh) -> bool:
    if getattr(mixer, "_agilerl_mamba_cp", False):
        return False
    cp = int(cp_mesh.size())
    group = cp_mesh.get_group()
    if getattr(mixer, "_agilerl_mamba_tp", False):
        tp = int(getattr(mixer, "_tp_degree", 1) or 1)
        _require_divisible(int(mixer.num_heads), tp * cp, "mamba_num_heads")
        _require_divisible(int(mixer.n_groups), tp * cp, "n_groups")
        _install_composed_mamba_cp(mixer, cp, group)
        return True
    _require_divisible(int(mixer.num_heads), cp, "mamba_num_heads")
    _require_divisible(int(mixer.n_groups), cp, "n_groups")
    original = mixer.forward
    local_heads = int(mixer.num_heads) // cp
    local_groups = int(mixer.n_groups) // cp
    local_inner = local_heads * int(mixer.head_dim)
    local_conv = local_inner + 2 * local_groups * int(mixer.ssm_state_size)

    def restore_full(self: nn.Module) -> None:
        saved = getattr(self, "_mamba_cp_saved", None)
        if saved is None:
            return
        nn.Module.__setattr__(self, "in_proj", saved["in_proj"])
        nn.Module.__setattr__(self, "out_proj", saved["out_proj"])
        _assign_view(self.conv1d, "weight", saved["conv_weight"])
        if saved["conv_bias"] is not None:
            _assign_view(self.conv1d, "bias", saved["conv_bias"])
        else:
            self.conv1d.bias = None
        _assign_view(self, "dt_bias", saved["dt_bias"])
        _assign_view(self, "A_log", saved["A_log"])
        _assign_view(self, "D", saved["D"])
        _assign_view(self.norm, "weight", saved["norm"])
        self.num_heads = saved["num_heads"]
        self.n_groups = saved["n_groups"]
        self.intermediate_size = saved["intermediate_size"]
        self.conv_dim = saved["conv_dim"]
        self.conv1d.in_channels = saved["conv_in"]
        self.conv1d.out_channels = saved["conv_out"]
        self.conv1d.groups = saved["conv_groups"]
        if saved["use_mem_eff_path"] is not None:
            self.use_mem_eff_path = saved["use_mem_eff_path"]
        object.__setattr__(self, "_mamba_cp_saved", None)

    def cp_forward(
        self: nn.Module, hidden_states: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        restore_full(self)
        rank = dist.get_rank(group)
        in_proj = self.in_proj
        out_proj = self.out_proj
        projected = in_proj(hidden_states)
        projected_hp = all_to_all_seq_to_hidden(projected, cp, group)
        slices = _slice_mamba_ssm(self, cp, rank)
        saved = {
            "in_proj": in_proj,
            "out_proj": out_proj,
            "conv_weight": self.conv1d.weight,
            "conv_bias": self.conv1d.bias,
            "dt_bias": self.dt_bias,
            "A_log": self.A_log,
            "D": self.D,
            "norm": self.norm.weight,
            "num_heads": self.num_heads,
            "n_groups": self.n_groups,
            "intermediate_size": self.intermediate_size,
            "conv_dim": self.conv_dim,
            "conv_in": self.conv1d.in_channels,
            "conv_out": self.conv1d.out_channels,
            "conv_groups": self.conv1d.groups,
            "use_mem_eff_path": getattr(self, "use_mem_eff_path", None),
        }
        # Register stand-ins in ``_modules`` so cuda_kernels_forward / torch_forward
        # call them. The fused scan reads ``out_proj.weight`` and skips that call.
        nn.Module.__setattr__(self, "in_proj", MixerInProjBypass(projected_hp, in_proj))
        nn.Module.__setattr__(self, "out_proj", MixerOutProjThenSeq(out_proj, cp, group))
        _assign_view(self.conv1d, "weight", slices["conv_weight"])
        if "conv_bias" in slices:
            _assign_view(self.conv1d, "bias", slices["conv_bias"])
        _assign_view(self, "dt_bias", slices["dt_bias"])
        _assign_view(self, "A_log", slices["A_log"])
        _assign_view(self, "D", slices["D"])
        _assign_view(self.norm, "weight", slices["norm"])
        self.num_heads = local_heads
        self.n_groups = local_groups
        self.intermediate_size = local_inner
        self.conv_dim = local_conv
        self.conv1d.in_channels = local_conv
        self.conv1d.out_channels = local_conv
        self.conv1d.groups = local_conv
        self.use_mem_eff_path = False
        object.__setattr__(self, "_mamba_cp_saved", saved)
        dummy = hidden_states.new_zeros(
            hidden_states.shape[0],
            projected_hp.shape[1],
            hidden_states.shape[-1],
        )
        inner = getattr(self, "torch_forward", original)
        self.forward = original
        try:
            return inner(dummy, *args, **kwargs)
        except Exception:
            restore_full(self)
            raise
        finally:
            self.forward = MethodType(cp_forward, self)

    mixer.forward = MethodType(cp_forward, mixer)
    object.__setattr__(mixer, "_agilerl_mamba_cp", True)
    return True


def install_gdn_context_parallel(mixer: nn.Module, cp_mesh: DeviceMesh) -> bool:
    if getattr(mixer, "_agilerl_gdn_cp", False):
        return False
    cp = int(cp_mesh.size())
    heads = getattr(mixer, "num_k_heads", None) or getattr(mixer, "num_heads", None)
    if heads is not None:
        _require_divisible(int(heads), cp, type(mixer).__name__ + " heads")
    n_groups = getattr(mixer, "n_groups", None)
    if n_groups is not None:
        _require_divisible(int(n_groups), cp, "n_groups")
    original = mixer.forward
    group = cp_mesh.get_group()

    def cp_forward(
        self: nn.Module, hidden_states: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        rank = dist.get_rank(group)
        full = gather_for_cp(hidden_states, group)
        self.forward = original
        try:
            out = original(full, *args, **kwargs)
        finally:
            self.forward = MethodType(cp_forward, self)
        return shard_for_cp(out, rank, cp)

    mixer.forward = MethodType(cp_forward, mixer)
    object.__setattr__(mixer, "_agilerl_gdn_cp", True)
    return True


def apply_mixer_context_parallel(model: nn.Module, cp_mesh: DeviceMesh | None) -> int:
    """Run each family's context-parallel mixer wrap on ``model``.

    :param model: Module tree that may contain mixers.
    :param cp_mesh: Context-parallel mesh. ``None`` or size 1 wraps nothing.
    :return: How many mixers were wrapped.
    """
    if cp_mesh is None or int(cp_mesh.size()) <= 1:
        return 0
    validate_cp_degree(int(cp_mesh.size()))
    return sum(plan.wrap(model, cp_mesh) for plan in family_context_parallel_plans())
