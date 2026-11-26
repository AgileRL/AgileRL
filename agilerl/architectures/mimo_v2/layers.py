# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Flex attention for ``MiMoV2Flash`` with the attention-sink factor taken from the fp32 log-sum-exp."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from torch.nn.attention.flex_attention import AuxOutput, AuxRequest, BlockMask
from transformers.integrations import flex_attention as flex_integration
from transformers.integrations.flex_attention import (
    compile_friendly_flex_attention,
    flex_attention_forward,
)
from transformers.modeling_utils import AttentionInterface
from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashAttention,
)

from agilerl.utils.patching import library_shim

__all__ = [
    "apply_attention_sinks",
    "install_mimo_v2_fp32_lse_flex_attention",
    "mimo_v2_flex_attention_forward",
]


def apply_attention_sinks(
    attention_output: torch.Tensor, lse: torch.Tensor, sinks: torch.Tensor
) -> torch.Tensor:
    """Renormalise attention output over the keys plus one sink logit per head.

    :param attention_output: Flex output, ``[batch, heads, seq_q, v_head_dim]``.
    :type attention_output: torch.Tensor
    :param lse: fp32 log-sum-exp of the scores, ``[batch, heads, seq_q]``.
    :type lse: torch.Tensor
    :param sinks: Sink logit per head, ``[heads]``.
    :type sinks: torch.Tensor
    :return: ``attention_output`` scaled by ``exp(lse - logsumexp(lse, sink))``,
        in its own dtype.
    :rtype: torch.Tensor
    """
    # exp(lse) / (exp(lse) + exp(sink)): the keys' share of the sink-augmented softmax.
    key_share = torch.sigmoid(lse.float() - sinks.float().view(1, -1, 1))
    return attention_output * key_share.to(attention_output.dtype).unsqueeze(-1)


def mimo_v2_flex_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | BlockMask,
    scaling: float | None = None,
    softcap: float | None = None,
    s_aux: torch.Tensor | None = None,
    position_bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run transformers' flex attention, applying MiMo-V2 sinks on the fp32 log-sum-exp.

    :param module: Attention module calling the backend.
    :type module: nn.Module
    :param query: Queries, ``[batch, heads, seq_q, head_dim]``.
    :type query: torch.Tensor
    :param key: Keys, ``[batch, kv_heads, seq_kv, head_dim]``.
    :type key: torch.Tensor
    :param value: Values, ``[batch, kv_heads, seq_kv, v_head_dim]``.
    :type value: torch.Tensor
    :param attention_mask: Flex block mask built by transformers' mask utilities.
    :type attention_mask: torch.Tensor | BlockMask
    :param scaling: Softmax scale.
    :type scaling: float | None
    :param softcap: Score softcap.
    :type softcap: float | None
    :param s_aux: Sink logit per head on sliding-window layers, else ``None``.
    :type s_aux: torch.Tensor | None
    :param position_bias: Additive score bias.
    :type position_bias: torch.Tensor | None
    :return: Attention output ``[batch, seq_q, heads, v_head_dim]`` and, from
        the stock forward only, its log-sum-exp.
    :rtype: tuple[torch.Tensor, torch.Tensor | None]
    """
    # The stock forward casts the lse to the value dtype before the sink
    # renormalisation, which loses precision in bf16; keep it only where no sink applies.
    if (
        s_aux is None
        or softcap is not None
        or position_bias is not None
        or not isinstance(module, MiMoV2FlashAttention)
    ):
        return flex_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            softcap=softcap,
            s_aux=s_aux,
            position_bias=position_bias,
            **kwargs,
        )
    # ``return_aux`` makes flex return the output and an AuxOutput holding the lse.
    attention_output, aux = cast(
        "tuple[torch.Tensor, AuxOutput]",
        compile_friendly_flex_attention(
            query,
            key,
            value,
            block_mask=attention_mask,
            enable_gqa=True,
            scale=scaling,
            kernel_options=kwargs.get("kernel_options"),
            training=module.training,
            return_aux=AuxRequest(lse=True),
        ),
    )
    lse = cast("torch.Tensor", aux.lse)
    attention_output = apply_attention_sinks(attention_output, lse, s_aux)
    return attention_output.transpose(1, 2).contiguous(), None


@library_shim("transformers", "==5.19.*")
def install_mimo_v2_fp32_lse_flex_attention() -> None:
    """Point transformers' ``flex_attention`` backend at :func:`mimo_v2_flex_attention_forward`."""
    # Must run before ``patch_flex_attention_kernel_options``, which wraps the
    # forward it imports at call time.
    integration: Any = flex_integration
    integration.flex_attention_forward = mimo_v2_flex_attention_forward
    AttentionInterface.register("flex_attention", mimo_v2_flex_attention_forward)
