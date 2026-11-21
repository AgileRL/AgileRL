# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Run the Qwen GatedDeltaNet output norm on fla's fused gated RMSNorm kernel."""

from __future__ import annotations

import importlib
import logging

import torch
from torch import nn

from agilerl.utils.patching import try_import

logger = logging.getLogger(__name__)

__all__ = ["GATED_NORM_TARGETS", "install_fused_gated_norm"]

GATED_NORM_TARGETS = (
    ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeRMSNormGated"),
    ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5RMSNormGated"),
)


def install_fused_gated_norm() -> None:
    """Point every Qwen GDN gated RMSNorm at fla's fused kernel when fla imports."""
    fused_norm_gate = try_import("fla.modules.fused_norm_gate")
    if fused_norm_gate is None:
        return
    rms_norm_gated = fused_norm_gate.rms_norm_gated

    def fused_gated_rms_norm_forward(
        self: nn.Module, hidden_states: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        return rms_norm_gated(
            hidden_states,
            gate,
            self.weight,
            None,
            activation=self.activation,
            eps=self.variance_epsilon,
        )

    for module_path, class_name in GATED_NORM_TARGETS:
        norm_cls = getattr(importlib.import_module(module_path), class_name)
        norm_cls.forward = fused_gated_rms_norm_forward
        logger.info("[qwen-gdn] %s runs fla's fused gated RMSNorm", class_name)
