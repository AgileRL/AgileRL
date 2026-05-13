"""LLM-specific fused-kernel ops (Liger and LoRA).

Public surface re-exported from sibling modules so callers can use either
``from agilerl.algorithms.core.llm_ops import LigerFusedLinearPolicyLossFunction``
or the fully-qualified submodule path.
"""

from agilerl.algorithms.core.llm_ops.fused_dpo_loss import _LigerDPOWithAlpha
from agilerl.algorithms.core.llm_ops.fused_lora import (
    clear_fused_adapter_routing,
    patch_lora_for_fused_forward,
    set_fused_adapter_routing,
)
from agilerl.algorithms.core.llm_ops.fused_policy_loss import (
    LigerFusedLinearPolicyLossFunction,
    _k3_kl,
    llm_policy_loss_fn,
)

__all__ = [
    "LigerFusedLinearPolicyLossFunction",
    "_LigerDPOWithAlpha",
    "_k3_kl",
    "clear_fused_adapter_routing",
    "llm_policy_loss_fn",
    "patch_lora_for_fused_forward",
    "set_fused_adapter_routing",
]
