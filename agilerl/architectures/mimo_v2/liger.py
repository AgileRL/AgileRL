# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Liger kernels for the native ``mimo_v2`` causal LM.

Its RMSNorm casts to the input dtype before the weight multiply, which is
Liger's ``llama`` casting mode, and each dense MLP computes ``silu(gate) * up``.
RoPE rotates only the first part of each head and routed experts run the
grouped expert kernels, so both keep their own forwards.
"""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from liger_kernel.ops import LigerSiLUMulFunction
from liger_kernel.transformers.monkey_patch import (
    MODEL_TYPE_TO_APPLY_LIGER_FN,
    _patch_rms_norm_module,
)
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from transformers.models.mimo_v2_flash import modeling_mimo_v2_flash

from agilerl.architectures.mimo_v2.configuration import MIMO_V2_MODEL_TYPE
from agilerl.utils.patching import library_shim

if TYPE_CHECKING:
    from transformers import PreTrainedModel

__all__ = [
    "apply_liger_kernel_to_mimo_v2",
    "dense_mlp_swiglu_forward",
    "register_mimo_v2_liger",
]

SWIGLU_ACTIVATIONS = frozenset({"silu", "swish"})


def dense_mlp_swiglu_forward(
    self: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """MiMo-V2 dense MLP with ``silu(gate) * up`` in Liger's SwiGLU kernel.

    :param hidden_states: ``(..., H)`` inputs.
    :type hidden_states: torch.Tensor
    :return: ``(..., H)`` outputs.
    :rtype: torch.Tensor
    """
    mlp: Any = self
    gated = LigerSiLUMulFunction.apply(
        mlp.gate_proj(hidden_states), mlp.up_proj(hidden_states)
    )
    return mlp.down_proj(gated)


@library_shim("liger-kernel", "==0.8.*")
@library_shim("transformers", "==5.19.*")
def apply_liger_kernel_to_mimo_v2(
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel | None = None,
) -> None:
    """Apply Liger RMSNorm and dense-MLP SwiGLU to native ``mimo_v2`` models.

    :param rms_norm: Patch RMSNorm with LigerRMSNorm.
    :type rms_norm: bool
    :param swiglu: Run each built dense MLP's ``silu(gate) * up`` in Liger's kernel.
    :type swiglu: bool
    :param model: Loaded model whose built norms and dense MLPs are also
        patched, or None.
    :type model: PreTrainedModel | None
    :return: None
    :rtype: None
    :raises ValueError: If ``swiglu`` is set and the model's activation is not SiLU.
    """
    mimo_mod: Any = modeling_mimo_v2_flash
    if rms_norm:
        mimo_mod.MiMoV2FlashRMSNorm = LigerRMSNorm
    if model is None:
        return
    text_config = model.config.get_text_config()
    if swiglu and text_config.hidden_act not in SWIGLU_ACTIVATIONS:
        msg = (
            "Liger SwiGLU needs a SiLU dense MLP; "
            f"{model.config.model_type} uses {text_config.hidden_act!r}"
        )
        raise ValueError(msg)
    base_model = model.get_decoder()
    if rms_norm:
        _patch_rms_norm_module(base_model.norm)
    for layer in base_model.layers:
        if rms_norm:
            _patch_rms_norm_module(layer.input_layernorm)
            _patch_rms_norm_module(layer.post_attention_layernorm)
        if swiglu and isinstance(layer.mlp, modeling_mimo_v2_flash.MiMoV2FlashMLP):
            layer.mlp.forward = MethodType(dense_mlp_swiglu_forward, layer.mlp)


def register_mimo_v2_liger() -> None:
    """Register ``mimo_v2`` with Liger's AutoLiger apply map.

    Liger's own applier wins if it ships one.

    :return: None
    :rtype: None
    """
    apply_fns: dict[str, Any] = MODEL_TYPE_TO_APPLY_LIGER_FN
    apply_fns.setdefault(MIMO_V2_MODEL_TYPE, apply_liger_kernel_to_mimo_v2)
