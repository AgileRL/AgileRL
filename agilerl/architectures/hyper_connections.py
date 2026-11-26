# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Compiled hyper-connection forward for DeepSeek-V4 and GLM-5.3.

Both families compute each call's weights the same way: an fp32 RMSNorm and
projection of the flattened ``[B, S, N, D]`` streams, a Sinkhorn-Knopp loop
(one column normalisation, then ``hc_sinkhorn_iters - 1`` row and column
pairs) for the ``[N, N]`` mixing matrix, and a ``pre``-weighted collapse.
All of it is one compiled graph that saves only the bf16 streams and
parameters; backward reruns the graph from them, so no fp32 intermediate
outlives the forward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

from agilerl.utils.patching import class_is_patched, library_shim

if TYPE_CHECKING:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4HyperConnection,
    )
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextHyperConnection,
    )

__all__ = [
    "COMPILED_HYPER_CONNECTION_MIX",
    "RecomputedHyperConnectionMix",
    "compile_hyper_connection_mix",
    "compiled_hyper_connection_mix",
    "hyper_connection_forward",
    "hyper_connection_mix",
    "sinkhorn_normalize",
]

COMPILED_HYPER_CONNECTION_MIX = "agilerl_compiled_hyper_connection_mix"


def sinkhorn_normalize(comb: torch.Tensor, iterations: int, eps: float) -> torch.Tensor:
    """Make each trailing ``[N, N]`` matrix doubly stochastic with Sinkhorn-Knopp.

    :param comb: ``[..., N, N]`` positive matrices.
    :param iterations: Column normalisations; rows are normalised one fewer time.
    :param eps: Added to every row and column sum.
    :return: The normalised matrices.
    """
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


def hyper_connection_mix(
    hidden_streams: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The stock hyper-connection maths in fp32, as a function of tensors.

    :param hidden_streams: ``[batch, seq, hc_mult, hidden]`` residual streams.
    :param fn: ``[(2 + hc_mult) * hc_mult, hc_mult * hidden]`` projection.
    :param base: Projection bias.
    :param scale: ``pre``, ``post`` and ``comb`` scales.
    :param hc_mult: Number of streams.
    :param sinkhorn_iters: Sinkhorn column normalisations.
    :param hc_eps: Hyper-connection epsilon.
    :param norm_eps: Epsilon of the unweighted input RMSNorm.
    :return: ``post`` and ``comb`` weights, and the streams collapsed by ``pre``.
    """
    batch_size, seq_len = hidden_streams.shape[:2]
    hc = hc_mult
    flattened = hidden_streams.view(batch_size, seq_len, -1).float()
    flattened = flattened * torch.rsqrt(
        flattened.square().mean(-1, keepdim=True) + norm_eps
    )
    flattened = F.linear(flattened, fn.float())
    pre_w, post_w, comb_w = flattened.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = base.split([hc, hc, hc * hc])
    pre_scale, post_scale, comb_scale = scale.unbind(0)
    comb_w = comb_w.view(*comb_w.shape[:-1], hc, hc)
    comb_b = comb_b.view(hc, hc)
    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + hc_eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb = torch.softmax(comb_w * comb_scale + comb_b, dim=-1) + hc_eps
    comb = sinkhorn_normalize(comb, sinkhorn_iters, hc_eps)
    collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
    return post, comb, collapsed


# Dynamic shapes, so a new batch or sequence length reuses the compiled graph.
compiled_hyper_connection_mix = torch.compile(
    hyper_connection_mix, fullgraph=True, dynamic=True
)


class RecomputedHyperConnectionMixCtx(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]
    needs_input_grad: tuple[bool, ...]
    scalars: tuple[int, int, float, float]

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class RecomputedHyperConnectionMix(torch.autograd.Function):
    """:func:`compiled_hyper_connection_mix` that saves only its inputs and reruns the graph in backward."""

    @staticmethod
    def forward(
        ctx: RecomputedHyperConnectionMixCtx,
        hidden_streams: torch.Tensor,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        hc_eps: float,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(hidden_streams, fn, base, scale)
        ctx.scalars = (hc_mult, sinkhorn_iters, hc_eps, norm_eps)
        return compiled_hyper_connection_mix(
            hidden_streams, fn, base, scale, *ctx.scalars
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx: RecomputedHyperConnectionMixCtx,
        grad_post: torch.Tensor,
        grad_comb: torch.Tensor,
        grad_collapsed: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        needs_grad = ctx.needs_input_grad[:4]
        with torch.enable_grad():
            leaves = [
                tensor.detach().requires_grad_(needs)
                for tensor, needs in zip(ctx.saved_tensors, needs_grad, strict=True)
            ]
            streams, fn, base, scale = leaves
            outputs = compiled_hyper_connection_mix(
                streams, fn, base, scale, *ctx.scalars
            )
        grads = iter(
            torch.autograd.grad(
                outputs,
                [leaf for leaf in leaves if leaf.requires_grad],
                (grad_post, grad_comb, grad_collapsed),
            )
        )
        input_grads = [next(grads) if needs else None for needs in needs_grad]
        return (*input_grads, None, None, None, None)


def hyper_connection_forward(
    self: DeepseekV4HyperConnection | Glm5NextTextHyperConnection,
    hidden_streams: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hyper-connection mixing weights from one compiled graph, saving only the bf16 streams and parameters.

    :param hidden_streams: ``[batch, seq, hc_mult, hidden]`` residual streams.
    :return: ``post`` and ``comb`` weights, and the streams collapsed by ``pre``.
    """
    return RecomputedHyperConnectionMix.apply(
        hidden_streams,
        self.fn,
        self.base,
        self.scale,
        self.hc_mult,
        self.hc_sinkhorn_iters,
        self.hc_eps,
        self.input_norm.eps,
    )


@library_shim("transformers", "==5.19.*")
def compile_hyper_connection_mix(
    hyper_connection_cls: type[nn.Module],
) -> None:
    """Install :func:`hyper_connection_forward` on a hyper-connection class once.

    :param hyper_connection_cls: ``DeepseekV4HyperConnection`` or ``Glm5NextTextHyperConnection``.
    """
    if class_is_patched(hyper_connection_cls, COMPILED_HYPER_CONNECTION_MIX):
        return
    hyper_connection_cls.forward = hyper_connection_forward
    setattr(hyper_connection_cls, COMPILED_HYPER_CONNECTION_MIX, True)
