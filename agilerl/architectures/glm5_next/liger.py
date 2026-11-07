# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Liger kernels for HuggingFace ``glm5_next`` and ``glm5_next_text``.

``Glm5NextTextRMSNorm`` casts to the input dtype before the weight multiply,
which is Liger's ``llama`` casting mode. Dense and shared-expert MLPs clamp
gate and up to ``swiglu_limit`` before ``silu(gate) * up``; the clamps stay in
PyTorch and the product runs in Liger's SwiGLU kernel. The DSA layers are NoPE,
so there is no RoPE to swap. Routed experts, the mHC norm, the gated
linear-attention norm, the indexer, and the vision tower keep their stock
forwards.
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
from transformers.models.glm5_next import modeling_glm5_next
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextMLP,
    Glm5NextTextModel,
    Glm5NextTextRMSNorm,
)

from agilerl.utils.patching import library_shim

if TYPE_CHECKING:
    from transformers import PreTrainedModel

__all__ = [
    "apply_liger_kernel_to_glm5_next",
    "clamped_swiglu_mlp_forward",
    "register_glm5_next_liger",
]

SWIGLU_ACTIVATIONS = frozenset({"silu", "swish"})
GLM5_NEXT_MODEL_TYPES = ("glm5_next", "glm5_next_text")


def clamped_swiglu_mlp_forward(
    self: Glm5NextTextMLP, hidden_states: torch.Tensor
) -> torch.Tensor:
    """``Glm5NextTextMLP.forward`` with ``silu(gate) * up`` in Liger's SwiGLU kernel.

    :param hidden_states: ``(..., H)`` inputs.
    :return: ``(..., H)`` outputs.
    """
    gate = self.gate_proj(hidden_states).clamp(min=None, max=self.swiglu_limit)
    up = self.up_proj(hidden_states).clamp(
        min=-self.swiglu_limit, max=self.swiglu_limit
    )
    return self.down_proj(LigerSiLUMulFunction.apply(gate, up))


@library_shim("liger-kernel", "==0.8.*")
@library_shim("transformers", "==5.19.*")
def apply_liger_kernel_to_glm5_next(
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel | None = None,
) -> None:
    """Apply Liger RMSNorm and dense / shared-expert SwiGLU to HuggingFace GLM-5.3 models.

    :param rms_norm: Patch the text RMSNorms with LigerRMSNorm.
    :param swiglu: Run each built dense and shared-expert MLP's ``silu(gate) * up`` in Liger's kernel.
    :param model: Loaded model whose built text modules are also patched, or None.
    :raises ValueError: If ``swiglu`` is set and the text MLPs are not SiLU.
    """
    glm_mod: Any = modeling_glm5_next
    if rms_norm:
        glm_mod.Glm5NextTextRMSNorm = LigerRMSNorm
    if model is None:
        return
    for text_model in (
        module for module in model.modules() if isinstance(module, Glm5NextTextModel)
    ):
        if swiglu and text_model.config.hidden_act not in SWIGLU_ACTIVATIONS:
            msg = (
                "Liger SwiGLU needs SiLU text MLPs; "
                f"{text_model.config.model_type} uses {text_model.config.hidden_act!r}"
            )
            raise ValueError(msg)
        for module in text_model.modules():
            if rms_norm and isinstance(module, Glm5NextTextRMSNorm):
                _patch_rms_norm_module(module)
            if swiglu and isinstance(module, Glm5NextTextMLP):
                mlp: Any = module
                mlp.forward = MethodType(clamped_swiglu_mlp_forward, mlp)


def register_glm5_next_liger() -> None:
    """Register both GLM-5.3 model types with Liger's AutoLiger map, keeping any applier Liger ships."""
    apply_fns: dict[str, Any] = MODEL_TYPE_TO_APPLY_LIGER_FN
    for model_type in GLM5_NEXT_MODEL_TYPES:
        apply_fns.setdefault(model_type, apply_liger_kernel_to_glm5_next)
