"""LLM-specific fused-kernel ops (Liger and LoRA).

The ``fused_lora`` helpers are always available; ``fused_loss``
(``LigerFusedLinearPolicyLossFunction``, ``LigerDPOWithAlpha``,
``llm_policy_loss_fn``) requires ``liger-kernel`` at import
time, so it is gated on :data:`agilerl.HAS_LIGER_KERNEL`. Without Liger
the public symbols resolve to ``None`` so callers' ``is None`` guard
fires.
"""

from agilerl import HAS_LIGER_KERNEL
from agilerl.algorithms.core.llm_ops.fused_lora import (
    patch_lora_for_fused_forward,
    set_fused_adapter_routing,
    unpatch_lora_for_fused_forward,
    unset_fused_adapter_routing,
)
from agilerl.algorithms.core.llm_ops.vllm_colocate import (
    get_vllm_internal_model,
    patch_vllm_lora_keep_resident,
    patch_vllm_strip_multimodal_towers,
)

if HAS_LIGER_KERNEL:
    from agilerl.algorithms.core.llm_ops.fused_loss import (
        LigerDPOWithAlpha,
        LigerFusedLinearPolicyLossFunction,
        apply_fused_policy_loss,
        llm_policy_loss_fn,
    )
else:
    LigerFusedLinearPolicyLossFunction = None  # type: ignore[assignment]
    LigerDPOWithAlpha = None  # type: ignore[assignment]
    apply_fused_policy_loss = None  # type: ignore[assignment]
    llm_policy_loss_fn = None  # type: ignore[assignment]

__all__ = [
    "LigerDPOWithAlpha",
    "LigerFusedLinearPolicyLossFunction",
    "apply_fused_policy_loss",
    "get_vllm_internal_model",
    "llm_policy_loss_fn",
    "patch_lora_for_fused_forward",
    "patch_vllm_lora_keep_resident",
    "patch_vllm_strip_multimodal_towers",
    "set_fused_adapter_routing",
    "unpatch_lora_for_fused_forward",
    "unset_fused_adapter_routing",
]
