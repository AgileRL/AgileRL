# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""MoE router forwards that renormalize top-k scores out of place.

Selective activation checkpointing saves each router's ``topk`` outputs, so a
router that renormalizes them in place (``/=``) mutates a saved tensor and the
backward recompute fails. These forwards keep each family's math and return
bitwise-equal outputs without writing to the ``topk`` result.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from agilerl.utils.patching import class_is_patched

logger = logging.getLogger(__name__)

__all__ = ["ROUTER_FORWARDS", "ROUTER_RENORM_OUT_OF_PLACE", "patch_moe_routers"]

ROUTER_RENORM_OUT_OF_PLACE = "agilerl_router_renorm_out_of_place"

RouterOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _softmax_topk(
    module: nn.Module, hidden_states: torch.Tensor, renormalize: bool
) -> RouterOutputs:
    """Qwen-style softmax top-k router, renormalizing into a new tensor."""
    router: Any = module
    hidden_states = hidden_states.reshape(-1, router.hidden_dim)
    router_logits = F.linear(hidden_states, router.weight)
    router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(router_probs, router.top_k, dim=-1)
    if renormalize:
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
    return router_logits, router_top_value.to(router_logits.dtype), router_indices


def renormalized_softmax_topk_forward(
    self: nn.Module, hidden_states: torch.Tensor
) -> RouterOutputs:
    """Softmax top-k router that always renormalizes (Qwen3.5-MoE, Qwen3-VL-MoE).

    :param hidden_states: Router input, flattened to ``(tokens, hidden)``.
    :type hidden_states: torch.Tensor
    :return: Router logits, top-k scores in the logits dtype, top-k expert indices.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    return _softmax_topk(self, hidden_states, renormalize=True)


def softmax_topk_forward(self: nn.Module, hidden_states: torch.Tensor) -> RouterOutputs:
    """Softmax top-k router that renormalizes when ``norm_topk_prob`` is set.

    :param hidden_states: Router input, flattened to ``(tokens, hidden)``.
    :type hidden_states: torch.Tensor
    :return: Router logits, top-k scores in the logits dtype, top-k expert indices.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    router: Any = self
    return _softmax_topk(self, hidden_states, renormalize=router.norm_topk_prob)


def mixtral_router_forward(
    self: nn.Module, hidden_states: torch.Tensor
) -> RouterOutputs:
    """Mixtral-style softmax top-k router that always renormalizes and keeps fp32 scores.

    :param hidden_states: Router input, flattened to ``(tokens, hidden)``.
    :type hidden_states: torch.Tensor
    :return: Router logits, fp32 top-k scores, top-k expert indices.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    router: Any = self
    hidden_states = hidden_states.reshape(-1, router.hidden_dim)
    router_logits = F.linear(hidden_states, router.weight)
    router_probs = F.softmax(router_logits.float(), dim=-1)
    router_top_value, router_indices = torch.topk(router_probs, router.top_k, dim=-1)
    router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
    return router_logits, router_top_value, router_indices


def gemma4_router_forward(
    self: nn.Module, hidden_states: torch.Tensor
) -> RouterOutputs:
    """Gemma4 MoE router: scaled RMS-normed projection, softmax top-k, expert scale.

    :param hidden_states: Router input of shape ``(..., hidden)``.
    :type hidden_states: torch.Tensor
    :return: fp32 router probabilities, scaled top-k weights, top-k expert indices.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    router: Any = self
    hidden_states = router.norm(hidden_states)
    hidden_states = hidden_states * router.scale * router.scalar_root_size
    expert_scores = router.proj(hidden_states)
    router_probabilities = F.softmax(expert_scores, dim=-1, dtype=torch.float32)
    top_k_weights, top_k_index = torch.topk(
        router_probabilities, k=router.config.top_k_experts, dim=-1
    )
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    top_k_weights = top_k_weights * router.per_expert_scale[top_k_index]
    return router_probabilities, top_k_weights, top_k_index


# Keyed by ``module.QualName`` so only classes a model holds are touched.
ROUTER_FORWARDS: Mapping[str, Callable[[nn.Module, torch.Tensor], RouterOutputs]] = {
    "transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeTopKRouter": (
        softmax_topk_forward
    ),
    "transformers.models.olmoe.modeling_olmoe.OlmoeTopKRouter": softmax_topk_forward,
    "transformers.models.flex_olmo.modeling_flex_olmo.FlexOlmoTopKRouter": (
        softmax_topk_forward
    ),
    "transformers.models.mellum.modeling_mellum.MellumTopKRouter": (
        softmax_topk_forward
    ),
    "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextTopKRouter": (
        softmax_topk_forward
    ),
    "transformers.models.mixtral.modeling_mixtral.MixtralTopKRouter": (
        mixtral_router_forward
    ),
    "transformers.models.minimax.modeling_minimax.MiniMaxTopKRouter": (
        mixtral_router_forward
    ),
    "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeTopKRouter": (
        softmax_topk_forward
    ),
    "transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextTopKRouter": (
        softmax_topk_forward
    ),
    "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextTopKRouter": (
        softmax_topk_forward
    ),
    "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.Qwen3OmniMoeTalkerTextTopKRouter": (
        softmax_topk_forward
    ),
    "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.Qwen3OmniMoeTextTopKRouter": (
        renormalized_softmax_topk_forward
    ),
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeTopKRouter": (
        renormalized_softmax_topk_forward
    ),
    "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextTopKRouter": (
        renormalized_softmax_topk_forward
    ),
    "transformers.models.gemma4.modeling_gemma4.Gemma4TextRouter": (
        gemma4_router_forward
    ),
}


def patch_moe_routers(model: nn.Module) -> None:
    """Install the out-of-place forward on each listed router class in *model*.

    :param model: Module whose submodules may include MoE routers.
    :type model: nn.Module
    """
    for module in model.modules():
        cls = type(module)
        forward = ROUTER_FORWARDS.get(f"{cls.__module__}.{cls.__qualname__}")
        if forward is None or class_is_patched(cls, ROUTER_RENORM_OUT_OF_PLACE):
            continue
        cls.forward = forward
        setattr(cls, ROUTER_RENORM_OUT_OF_PLACE, True)
        logger.info("[moe-router] %s renormalizes scores out of place", cls.__name__)
