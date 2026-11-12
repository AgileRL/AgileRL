# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the gloo tensor-, context- and expert-parallel tests."""

from __future__ import annotations

import copy
import importlib.machinery
import itertools
import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn.attention.flex_attention import (
    AuxOutput,
    AuxRequest,
    BlockMask,
    create_mask,
)
from torch.utils._python_dispatch import TorchDispatchMode

from agilerl.distributed import (
    FSDPConfig,
    ParallelMesh,
    build_parallel_mesh,
    fsdp_blocks,
)
from agilerl.distributed.context_parallel import (
    substitute_hf_ulysses_attn,
    update_ulysses_params,
)
from agilerl.distributed.fsdp import (
    materialize_dtensors,
    materialize_fsdp2_from_cpu_state,
)
from agilerl.distributed.mesh import MeshAxisName
from agilerl.distributed.runtime import FSDPRuntime
from agilerl.distributed.tensor_parallel import (
    tensor_parallel_row_orders,
    tensor_rows_in_order,
)
from agilerl.utils.llm_utils import get_lora_named_params

# Eager and functional collectives, by the name both kinds are recorded under.
COLLECTIVE_KINDS = {
    ("c10d", "alltoall_base_"): "all_to_all",
    ("c10d", "allreduce_"): "all_reduce",
    ("c10d", "allgather_"): "all_gather",
    ("_c10d_functional", "all_to_all_single"): "all_to_all",
    ("_c10d_functional", "all_reduce"): "all_reduce",
    ("_c10d_functional", "all_gather_into_tensor"): "all_gather",
}


class CollectiveRecorder(TorchDispatchMode):
    """Record every eager or functional collective dispatched under it, with its input."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, bool, tuple[int, ...], torch.dtype]] = []

    def __torch_dispatch__(
        self,
        func: torch._ops.OpOverload,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kind = COLLECTIVE_KINDS.get((func.namespace, func.overloadpacket.__name__))
        if kind is not None:
            functional = func.namespace == "_c10d_functional"
            if functional:
                tensor = args[0]
            elif kind == "all_to_all":
                tensor = args[1]
            else:
                tensor = args[0][0] if kind == "all_reduce" else args[1][0]
            self.calls.append((kind, functional, tuple(tensor.shape), tensor.dtype))
        return func(*args, **(kwargs or {}))

    @property
    def kinds(self) -> list[str]:
        """Collective kinds in dispatch order."""
        return [kind for kind, _, _, _ in self.calls]

    def rows(self, kind: str) -> int:
        """Floating-point rows sent through collectives of ``kind``."""
        return sum(
            shape[0]
            for op, _, shape, dtype in self.calls
            if op == kind and dtype.is_floating_point
        )


def assert_exact(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    # Sharded fp32 sums reduce in another order, so a near-zero element can be
    # off by a few ulps of the tensor's largest element.
    assert actual.shape == expected.shape, label
    if expected.numel() == 0:
        return
    scale = max(expected.abs().max().item(), 1.0)
    if not torch.allclose(actual, expected, atol=1e-5 * scale, rtol=1e-5):
        diff = (actual - expected).abs().max().item()
        msg = f"{label}: max abs diff {diff}"
        raise AssertionError(msg)


def broadcast_module(module: nn.Module) -> None:
    for param in module.parameters():
        dist.broadcast(param.data, src=0)


def broadcast_tensor(tensor: torch.Tensor) -> torch.Tensor:
    dist.broadcast(tensor, src=0)
    return tensor


def with_lora(module: nn.Module, targets: list[str], std: float = 0.5) -> nn.Module:
    inject_adapter_in_model(LoraConfig(r=2, target_modules=targets), module)
    module.requires_grad_(True)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "lora_" in name:
                param.normal_(std=std)
    return module


def unwrap(module: nn.Module) -> nn.Module:
    inner = getattr(module, "_checkpoint_wrapped_module", None)
    if isinstance(inner, nn.Module):
        return unwrap(inner)
    return module


def plain(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


def on_tp_mesh(param: DTensor) -> bool:
    names = param.device_mesh.mesh_dim_names or ()
    return MeshAxisName.TP in names


def on_ep_mesh(param: DTensor) -> bool:
    return MeshAxisName.EP in (param.device_mesh.mesh_dim_names or ())


def world_held_numel(param: DTensor) -> int:
    held = torch.tensor(param.to_local().numel())
    dist.all_reduce(held)
    return int(held)


def assert_same_state(got: Any, expected: Any, where: str) -> None:
    if isinstance(expected, dict):
        assert got.keys() == expected.keys(), where
        for key, value in expected.items():
            assert_same_state(got[key], value, f"{where}.{key}")
    elif isinstance(expected, list):
        assert len(got) == len(expected), where
        for index, (left, right) in enumerate(zip(got, expected, strict=True)):
            assert_same_state(left, right, f"{where}[{index}]")
    elif isinstance(expected, torch.Tensor):
        assert torch.equal(got, expected), where
    else:
        assert got == expected, where


def assert_lora(model: nn.Module, reference: nn.Module) -> None:
    """Gathered adapters, in HF row order, equal the unsharded ones.

    EP keeps expert B as ``[out, r, E]``.
    """
    named = get_lora_named_params(model)
    orders = tensor_parallel_row_orders(model)
    reference_params = {
        plain(name): param for name, param in reference.named_parameters()
    }
    with materialize_dtensors(*[param for _, param in named]) as dense:
        for (name, _), gathered in zip(named, dense, strict=True):
            order = orders.get(name)
            exported = tensor_rows_in_order(
                gathered, None if order is None else torch.argsort(order)
            )
            expected = reference_params[plain(name)].detach()
            assert_exact(exported, expected.reshape(exported.shape), name)


def assert_resharded(
    model: nn.Module, reference: nn.Module, tp: int, fsdp_size: int
) -> None:
    """Each parameter is split across ranks again; TP weights stay on the TP mesh."""
    reference_params = {
        plain(name): param for name, param in reference.named_parameters()
    }
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), name
        full = reference_params[plain(name)]
        held = world_held_numel(param)
        if on_ep_mesh(param):
            # Expert LoRA shards on ep only and is replicated over edp_shard.
            split = 1
            for dim, placement in enumerate(param.placements):
                if not placement.is_replicate():
                    split *= int(param.device_mesh.size(dim))
            assert held == full.numel() * (dist.get_world_size() // split), name
        elif on_tp_mesh(param):
            assert param.device_mesh.mesh_dim_names[-1] == MeshAxisName.TP, name
            assert param.device_mesh.size() == tp * fsdp_size, name
            copies = tp if param.placements[-1].is_replicate() else 1
            assert held == full.numel() * copies, name
        else:
            assert param.device_mesh.size() == fsdp_size, name
            assert held == full.numel() * tp, name


def parallel_mesh(world_size: int, cp: int, tp: int, ep: int) -> ParallelMesh:
    mesh = build_parallel_mesh(
        world_size=world_size, ep=ep, tp=tp, cp=cp, device_type="cpu"
    )
    if mesh is not None:
        return mesh
    # fully_shard without a mesh uses the accelerator, not this gloo group.
    return ParallelMesh(
        dp_shard=init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp_shard",))
    )


def fsdp_config(
    cp: int,
    tp: int,
    ep: int,
    dtype: str = "float32",
    shard_frozen_tp: bool | None = None,
    reshard_after_forward: bool = True,
) -> FSDPConfig:
    return FSDPConfig(
        reshard_after_forward=reshard_after_forward,
        ep=ep,
        tp=tp,
        cp=cp,
        param_dtype=dtype,
        reduce_dtype=dtype,
        routed_expert_chunk_mib=64,
        checkpoint_every_n_blocks=1,
        shard_frozen_tp=shard_frozen_tp,
    )


def sgd(model: nn.Module, lr: float) -> SimpleNamespace:
    by_mesh: dict[Any, list[nn.Parameter]] = {}
    for param in model.parameters():
        if param.requires_grad:
            key = param.device_mesh if isinstance(param, DTensor) else None
            by_mesh.setdefault(key, []).append(param)
    optimizer = torch.optim.SGD(
        [{"params": group} for group in by_mesh.values()], lr=lr
    )
    return SimpleNamespace(
        _single_optimizer=lambda: optimizer,
        step=optimizer.step,
        zero_grad=optimizer.zero_grad,
    )


def lora_runtime(
    dense: nn.Module,
    world_size: int,
    cp: int,
    tp: int,
    ep: int,
    dtype: str = "float32",
    shard_frozen_tp: bool | None = None,
    reshard_after_forward: bool = True,
) -> tuple[nn.Module, FSDPRuntime, SimpleNamespace]:
    """Shard a copy of ``dense`` and pair it with an AdamW over its LoRA params."""
    stack = copy.deepcopy(dense)
    mesh = parallel_mesh(world_size, cp, tp, ep)
    config = fsdp_config(cp, tp, ep, dtype, shard_frozen_tp, reshard_after_forward)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(stack, "cpu", config, parallel_mesh=mesh)
    runtime = FSDPRuntime(config)
    runtime.parallel_mesh = mesh
    by_mesh: dict[Any, list[nn.Parameter]] = {}
    for param in stack.parameters():
        if param.requires_grad:
            key = param.device_mesh if isinstance(param, DTensor) else None
            by_mesh.setdefault(key, []).append(param)
    adam = torch.optim.AdamW([{"params": group} for group in by_mesh.values()], lr=1e-2)
    optimizer = SimpleNamespace(
        _single_optimizer=lambda: adam, step=adam.step, zero_grad=adam.zero_grad
    )
    return stack, runtime, optimizer


def cpu_flash_attn_varlen_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    _cu_seqlens_k: torch.Tensor,
    _max_seqlen_q: int,
    _max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
    **_kwargs: Any,
) -> torch.Tensor:
    """Causal varlen attention as dense math; flash-attn has no CPU build."""
    assert causal
    bounds = cu_seqlens_q.tolist()
    outputs = []
    for start, stop in itertools.pairwise(bounds):
        q = query[start:stop].transpose(0, 1)
        k = key[start:stop].transpose(0, 1)
        v = value[start:stop].transpose(0, 1)
        k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)
        v = v.repeat_interleave(q.shape[0] // v.shape[0], dim=0)
        scores = (q @ k.transpose(-1, -2)) * softmax_scale
        length = stop - start
        future = torch.ones(length, length, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
        outputs.append((torch.softmax(scores, dim=-1) @ v).transpose(0, 1))
    return torch.cat(outputs)


def publish_varlen_offsets(cu_seqlens: torch.Tensor) -> None:
    """Publish CPU ``cu_seqlens`` and their host copy as the Ulysses offsets."""
    update_ulysses_params(cu_seqlens, tuple(cu_seqlens.tolist()))


def install_cpu_flash_attn() -> None:
    """Register :func:`cpu_flash_attn_varlen_func` as this process's ``flash_attn`` package."""
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    flash_attn.flash_attn_varlen_func = cpu_flash_attn_varlen_func
    sys.modules["flash_attn"] = flash_attn


def use_ulysses_flash(config: Any, cp_group: dist.ProcessGroup) -> None:
    """Route attention under ``config`` through Ulysses over the CPU flash stand-in."""
    install_cpu_flash_attn()
    config._attn_implementation = "flash_attention_2"
    substitute_hf_ulysses_attn(cp_group, "flash_attention_2")


def cpu_flex_attention(
    _module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: BlockMask,
    scaling: float,
    s_aux: torch.Tensor | None = None,
    **_kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Flex attention with sinks as dense math over the BlockMask; CPU flex has no backward."""
    _, heads, q_len, _ = query.shape
    allowed = create_mask(
        attention_mask.mask_mod, None, None, q_len, key.shape[-2], query.device
    )
    key = key.repeat_interleave(heads // key.shape[1], dim=1)
    value = value.repeat_interleave(heads // value.shape[1], dim=1)
    scores = (query @ key.transpose(-1, -2)) * scaling
    scores = scores.masked_fill(~allowed, float("-inf"))
    if s_aux is not None:
        sinks = s_aux.view(1, -1, 1, 1).expand(*scores.shape[:-1], 1)
        scores = torch.cat([scores, sinks], dim=-1)
    probs = torch.softmax(scores, dim=-1)[..., : key.shape[-2]]
    return (probs @ value).transpose(1, 2).contiguous(), None


def dense_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: BlockMask,
    scale: float,
    return_aux: AuxRequest | None = None,
    **_flex_options: Any,
) -> torch.Tensor | tuple[torch.Tensor, AuxOutput]:
    """``compile_friendly_flex_attention`` as dense math over the BlockMask; CPU flex has no backward."""
    batch, heads, q_len, _ = query.shape
    allowed = create_mask(
        block_mask.mask_mod, batch, 1, q_len, key.shape[-2], query.device
    )
    key = key.repeat_interleave(heads // key.shape[1], dim=1)
    value = value.repeat_interleave(heads // value.shape[1], dim=1)
    scores = (query @ key.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~allowed, float("-inf"))
    output = torch.softmax(scores, dim=-1) @ value
    if return_aux is None:
        return output
    return output, AuxOutput(lse=torch.logsumexp(scores, dim=-1))
