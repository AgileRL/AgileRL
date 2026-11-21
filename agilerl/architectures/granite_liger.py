# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Liger kernels for HuggingFace ``granitemoe`` and ``granitemoehybrid``.

Both RMSNorms cast to the input dtype before the weight multiply, which is
Liger's ``llama`` casting mode, and both RoPEs are the Llama rotation. The
hybrid shared MLP computes ``silu(gate) * up`` on the halves of one fused
projection. Routed experts and the Mamba2 gated norm keep their stock
forwards.
"""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any, cast

import torch
from liger_kernel.ops import LigerSiLUMulFunction
from liger_kernel.transformers.monkey_patch import (
    MODEL_TYPE_TO_APPLY_LIGER_FN,
    _patch_rms_norm_module,
)
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from transformers.models.granitemoe import modeling_granitemoe
from transformers.models.granitemoehybrid import modeling_granitemoehybrid

from agilerl.utils.patching import library_shim

if TYPE_CHECKING:
    from transformers import PreTrainedModel

__all__ = [
    "apply_liger_kernel_to_granitemoe",
    "apply_liger_kernel_to_granitemoehybrid",
    "register_granite_liger",
    "shared_mlp_swiglu_forward",
]

SWIGLU_ACTIVATIONS = frozenset({"silu", "swish"})


def shared_mlp_swiglu_forward(
    self: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """GraniteMoeHybrid shared MLP with ``silu(gate) * up`` in Liger's SwiGLU kernel.

    :param hidden_states: ``(..., H)`` inputs.
    :type hidden_states: torch.Tensor
    :return: ``(..., H)`` outputs.
    :rtype: torch.Tensor
    """
    mlp: Any = self
    gate, up = mlp.input_linear(hidden_states).chunk(2, dim=-1)
    return mlp.output_linear(LigerSiLUMulFunction.apply(gate, up))


def _patch_decoder_norms(base_model: torch.nn.Module) -> None:
    """Bind Liger RMSNorm onto the final norm and every decoder layer's two norms."""
    _patch_rms_norm_module(base_model.norm)
    for layer in cast("list[Any]", base_model.layers):
        _patch_rms_norm_module(layer.input_layernorm)
        _patch_rms_norm_module(layer.post_attention_layernorm)


@library_shim("liger-kernel", "==0.8.*")
@library_shim("transformers", "==5.19.*")
def apply_liger_kernel_to_granitemoe(
    rope: bool = True,
    rms_norm: bool = True,
    model: PreTrainedModel | None = None,
) -> None:
    """Apply Liger RoPE and RMSNorm to HuggingFace ``granitemoe`` models.

    :param rope: Patch rotary embedding with Liger's kernel.
    :type rope: bool
    :param rms_norm: Patch RMSNorm with LigerRMSNorm.
    :type rms_norm: bool
    :param model: Loaded model whose built norms are also patched, or None.
    :type model: PreTrainedModel | None
    :return: None
    :rtype: None
    """
    granite_mod: Any = modeling_granitemoe
    if rope:
        granite_mod.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        granite_mod.GraniteMoeRMSNorm = LigerRMSNorm
    if model is not None and rms_norm:
        _patch_decoder_norms(getattr(model, model.base_model_prefix))


@library_shim("liger-kernel", "==0.8.*")
@library_shim("transformers", "==5.19.*")
def apply_liger_kernel_to_granitemoehybrid(
    rope: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel | None = None,
) -> None:
    """Apply Liger RoPE, RMSNorm, and shared-MLP SwiGLU to HuggingFace ``granitemoehybrid`` models.

    :param rope: Patch rotary embedding with Liger's kernel.
    :type rope: bool
    :param rms_norm: Patch RMSNorm with LigerRMSNorm.
    :type rms_norm: bool
    :param swiglu: Run each built shared MLP's ``silu(gate) * up`` in Liger's kernel.
    :type swiglu: bool
    :param model: Loaded model whose built norms and shared MLPs are also
        patched, or None.
    :type model: PreTrainedModel | None
    :return: None
    :rtype: None
    :raises ValueError: If ``swiglu`` is set and the model's activation is not SiLU.
    """
    granite_mod: Any = modeling_granitemoehybrid
    if rope:
        granite_mod.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        granite_mod.GraniteMoeHybridRMSNorm = LigerRMSNorm
    if model is None:
        return
    if swiglu and model.config.hidden_act not in SWIGLU_ACTIVATIONS:
        msg = (
            "Liger SwiGLU needs a SiLU shared MLP; "
            f"{model.config.model_type} uses {model.config.hidden_act!r}"
        )
        raise ValueError(msg)
    base_model = getattr(model, model.base_model_prefix)
    if rms_norm:
        _patch_decoder_norms(base_model)
    if swiglu:
        for layer in base_model.layers:
            layer.shared_mlp.forward = MethodType(
                shared_mlp_swiglu_forward, layer.shared_mlp
            )


def register_granite_liger() -> None:
    """Register both Granite model types with Liger's AutoLiger apply map.

    Liger's own applier wins if it ships one for either type.

    :return: None
    :rtype: None
    """
    apply_fns: dict[str, Any] = MODEL_TYPE_TO_APPLY_LIGER_FN
    apply_fns.setdefault("granitemoe", apply_liger_kernel_to_granitemoe)
    apply_fns.setdefault("granitemoehybrid", apply_liger_kernel_to_granitemoehybrid)
