# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Packed mixture-of-experts layouts: detection and expert activations."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from agilerl import HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES
from agilerl.architectures.gptoss import layers as gptoss_layers
from agilerl.lora.moe.grouped_gemm import ExpertKernels

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from transformers.activations import SiLUActivation, get_activation
    from transformers.integrations.moe import _default_apply_gate
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4Experts,
    )
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextExperts,
    )

if HAS_LIGER_KERNEL:
    from liger_kernel.transformers.swiglu import LigerExperts

is_transposed_experts_module = gptoss_layers.is_gpt_oss_experts_module


def _forward_param_names(module: nn.Module) -> list[str]:
    """Positional parameter names of a module's ``forward``, excluding ``self``."""
    try:
        signature = inspect.signature(type(module).forward)
    except (TypeError, ValueError):
        return []
    return [name for name in signature.parameters if name != "self"]


def is_sorted_experts_module(module: nn.Module) -> bool:
    """Whether *module* is a grouped linear over expert-sorted rows with a stacked 3D ``weight``."""
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 3:
        return False
    return _forward_param_names(module)[:2] == ["inputs", "expert_size"]


def routed_projection_names(module: nn.Module) -> tuple[str, bool] | None:
    """The up-projection parameter name and gatedness of a self-routing packed-experts block.

    Gated (``gate_up_proj``, Qwen3-MoE/granite: ``act(gate) * up``) and
    ungated (``up_proj``, NemotronH: ``act(up)``) variants are supported;
    per-expert biases or other layouts are off-convention and return ``None``.
    """
    down = getattr(module, "down_proj", None)
    if not isinstance(down, torch.Tensor) or down.ndim != 3:
        return None
    if getattr(module, "down_proj_bias", None) is not None:
        return None
    if _forward_param_names(module)[:3] != [
        "hidden_states",
        "top_k_index",
        "top_k_weights",
    ]:
        return None
    for up_name, gated in (("gate_up_proj", True), ("up_proj", False)):
        up = getattr(module, up_name, None)
        if not isinstance(up, torch.Tensor) or up.ndim != 3:
            continue
        if getattr(module, f"{up_name}_bias", None) is not None:
            continue
        num_experts, up_out, in_dim = up.shape
        if gated and up_out % 2:
            continue
        intermediate = up_out // 2 if gated else up_out
        if down.shape == (num_experts, in_dim, intermediate):
            return up_name, gated
    return None


def is_routed_experts_module(module: nn.Module) -> bool:
    """Whether *module* is a self-routing packed-experts block."""
    return routed_projection_names(module) is not None


def is_packed_experts_module(module: nn.Module) -> bool:
    """Whether *module* is a packed expert stack (routed, sorted, or transposed)."""
    return (
        is_routed_experts_module(module)
        or is_sorted_experts_module(module)
        or is_transposed_experts_module(module)
    )


def routed_experts_act_fn(
    experts: nn.Module,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Activation for the packed grouped-GEMM path; never guessed.

    Stock HF packed experts expose ``act_fn`` or a config with ``hidden_act``.
    ``LigerExperts`` keeps neither but only builds for silu, so it runs SiLU.
    Anything else raises instead of assuming a default.
    """
    act_fn = getattr(experts, "act_fn", None)
    if callable(act_fn):
        return act_fn
    if HAS_LIGER_KERNEL and isinstance(experts, LigerExperts):
        return F.silu
    config = getattr(experts, "config", None)
    hidden_act = getattr(config, "hidden_act", None)
    resolved = get_activation(hidden_act) if isinstance(hidden_act, str) else None
    if resolved is None:
        msg = (
            "Packed-experts module does not expose an activation function. "
            "Set ``act_fn`` on the module or provide ``config.hidden_act``."
        )
        raise RuntimeError(msg)
    return resolved


@dataclass(frozen=True)
class ClampedSwiGLU:
    """An HF ``_apply_gate`` that computes ``silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)``."""

    apply_gate: Callable[[torch.Tensor], torch.Tensor]
    limit: float

    def __call__(self, projected: torch.Tensor) -> torch.Tensor:
        return self.apply_gate(projected)


def routed_experts_gate(
    experts: nn.Module,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """The experts' own HF ``_apply_gate`` on the packed ``[gate | up]`` projection, or ``None`` for ``act_fn(gate) * up``.

    :param experts: Packed routed-experts module.
    :type experts: nn.Module
    :return: The bound override, as a :class:`ClampedSwiGLU` when it is a
        known clamped SiLU gate, or ``None`` when the class keeps HF's default.
    :rtype: Callable[[torch.Tensor], torch.Tensor] | None
    """
    apply_gate = getattr(type(experts), "_apply_gate", None)
    if apply_gate is None or apply_gate is _default_apply_gate:
        return None
    bound = MethodType(apply_gate, experts)
    # HF classes whose gate clamps a SiLU gate, and the attribute holding the limit.
    limit_attrs = {DeepseekV4Experts: "limit", Glm5NextTextExperts: "swiglu_limit"}
    gate_owner = next(
        cls for cls in type(experts).__mro__ if "_apply_gate" in vars(cls)
    )
    limit_attr = limit_attrs.get(gate_owner)
    # DeepSeek-V4 applies its configured ``act_fn``; GLM-5.3 calls F.silu.
    act_fn = getattr(experts, "act_fn", F.silu)
    if limit_attr is None or not is_silu(act_fn):
        return bound
    return ClampedSwiGLU(bound, float(getattr(experts, limit_attr)))


def is_silu(act_fn: Callable[[torch.Tensor], torch.Tensor]) -> bool:
    """Whether *act_fn* is SiLU."""
    return act_fn is F.silu or isinstance(act_fn, (nn.SiLU, SiLUActivation))


class SwiGLUCtx(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]
    kernels: ExpertKernels
    limit: float | None

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class SwiGLU(torch.autograd.Function):
    """``silu(gate) * up`` of a packed ``[gate | up]`` projection on ``kernels``, optionally clamped to ``limit``; saves only the projection."""

    @staticmethod
    def forward(
        ctx: SwiGLUCtx,
        projected: torch.Tensor,
        kernels: ExpertKernels,
        limit: float | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(projected)
        ctx.kernels = kernels
        ctx.limit = limit
        return kernels.swiglu(projected, limit)

    @staticmethod
    @once_differentiable
    def backward(ctx: SwiGLUCtx, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (projected,) = ctx.saved_tensors
        return ctx.kernels.swiglu_backward(grad, projected, ctx.limit), None, None


class RecomputedGateCtx(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]
    apply_gate: Callable[[torch.Tensor], torch.Tensor]

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class RecomputedGate(torch.autograd.Function):
    """An experts family's own gate on a packed projection; saves only the projection and reruns the gate in backward."""

    @staticmethod
    def forward(
        ctx: RecomputedGateCtx,
        projected: torch.Tensor,
        apply_gate: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        ctx.save_for_backward(projected)
        ctx.apply_gate = apply_gate
        return apply_gate(projected)

    @staticmethod
    @once_differentiable
    def backward(
        ctx: RecomputedGateCtx, grad: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (projected,) = ctx.saved_tensors
        with torch.enable_grad():
            leaf = projected.detach().requires_grad_(True)
            out = ctx.apply_gate(leaf)
        (grad_projected,) = torch.autograd.grad(out, leaf, grad)
        return grad_projected, None


def expert_activation(
    projected: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    gated: bool,
    kernels: ExpertKernels | None,
    apply_gate: Callable[[torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    """``act(gate) * up`` for a gated up-projection, else ``act(up)``.

    :param kernels: Expert kernels that run a gated SiLU, clamped or not, as
        one fused :class:`SwiGLU`, or ``None`` for torch ops.
    :param apply_gate: The family's own gate from :func:`routed_experts_gate`,
        which replaces ``act(gate) * up``, or ``None``.
    """
    if kernels is not None and isinstance(apply_gate, ClampedSwiGLU):
        return SwiGLU.apply(projected, kernels, apply_gate.limit)
    if apply_gate is not None:
        return RecomputedGate.apply(projected, apply_gate)
    if gated and kernels is not None and is_silu(act_fn):
        return SwiGLU.apply(projected, kernels, None)
    if gated:
        gate, up = projected.chunk(2, dim=-1)
        return act_fn(gate) * up
    return act_fn(projected)
