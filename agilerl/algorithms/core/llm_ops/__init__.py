"""LLM-specific fused-kernel ops (Liger and LoRA).

The ``fused_lora`` helpers are always available; ``fused_loss``
(``LigerFusedLinearPolicyLossFunction``, ``_LigerDPOWithAlpha``,
``llm_policy_loss_fn``) requires ``liger-kernel`` at import
time, so it is gated on :data:`agilerl.HAS_LIGER_KERNEL`. Without Liger
the public symbols resolve to ``None`` so callers' ``is None`` guard
fires.
"""

from agilerl import HAS_LIGER_KERNEL
from agilerl.algorithms.core.llm_ops.fused_lora import (
    clear_fused_adapter_routing,
    patch_lora_for_fused_forward,
    set_fused_adapter_routing,
)

if HAS_LIGER_KERNEL:
    from agilerl.algorithms.core.llm_ops.fused_loss import (
        LigerFusedLinearPolicyLossFunction,
        _LigerDPOWithAlpha,
        llm_policy_loss_fn,
    )
else:
    LigerFusedLinearPolicyLossFunction = None  # type: ignore[assignment]
    _LigerDPOWithAlpha = None  # type: ignore[assignment]
    llm_policy_loss_fn = None  # type: ignore[assignment]

__all__ = [
    "LigerFusedLinearPolicyLossFunction",
    "_LigerDPOWithAlpha",
    "clear_fused_adapter_routing",
    "llm_policy_loss_fn",
    "patch_lora_for_fused_forward",
    "set_fused_adapter_routing",
]
