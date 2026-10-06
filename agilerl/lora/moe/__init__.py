# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Split low-rank adapter execution for packed mixture-of-experts weights.

PEFT's ``ParamWrapper`` (``LoraConfig.target_parameters``) supports stacked 3D
expert weights but applies adapters by materializing the full-rank delta
``B @ A`` for every expert on every forward — an allocation the size of the
expert weights themselves, per wrapped parameter, per layer. The wrappers here
keep the low-rank factorization split instead: tokens are grouped per expert
and pushed through that expert's rank-``r`` slice of ``lora_A``/``lora_B``, so
the largest adapter intermediate is ``[tokens, r]``. Wrappers on modules
matching none of the supported calling conventions stay on PEFT's default path.

``layouts`` detects packed-expert calling conventions, ``grouped_gemm`` holds
the per-expert GEMMs, ``adapters`` the LoRA deltas, ``routed`` / ``recompute``
/ ``transposed`` the expert forwards, ``wrappers`` the ``ParamWrapper``
subclasses, and ``model`` the whole-model passes.
"""

from agilerl.lora.moe.adapters import resolve_adapters, split_lora_delta, wrapper_chain
from agilerl.lora.moe.grouped_gemm import grouped_mm_supported
from agilerl.lora.moe.model import (
    bind_routed_experts_config,
    install_packed_expert_grouped_gemm,
    moe_expert_target_parameters,
    set_routed_experts_recompute,
    upgrade_moe_param_wrappers,
)
from agilerl.lora.moe.recompute import LoraExpertsFunction
from agilerl.lora.moe.routed import routed_experts_local_forward
from agilerl.lora.moe.transposed import transposed_experts_local_forward
from agilerl.lora.moe.wrappers import (
    RoutedExpertsLoraWrapper,
    SortedExpertsLoraWrapper,
    TransposedExpertsLoraWrapper,
)

__all__ = [
    "LoraExpertsFunction",
    "RoutedExpertsLoraWrapper",
    "SortedExpertsLoraWrapper",
    "TransposedExpertsLoraWrapper",
    "bind_routed_experts_config",
    "grouped_mm_supported",
    "install_packed_expert_grouped_gemm",
    "moe_expert_target_parameters",
    "resolve_adapters",
    "routed_experts_local_forward",
    "set_routed_experts_recompute",
    "split_lora_delta",
    "transposed_experts_local_forward",
    "upgrade_moe_param_wrappers",
    "wrapper_chain",
]
