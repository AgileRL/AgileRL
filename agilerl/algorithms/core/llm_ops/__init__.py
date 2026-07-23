"""LLM-specific fused-kernel ops (Liger and LoRA).

The ``fused_lora`` helpers wrap PEFT LoRA layers, so they are gated on
:data:`agilerl.HAS_LLM_DEPENDENCIES`; ``fused_loss``
(``LigerFusedLinearPolicyLossFunction``, ``LigerDPOWithAlpha``,
``llm_policy_loss_fn``) requires ``liger-kernel`` at import
time, so it is gated on :data:`agilerl.HAS_LIGER_KERNEL`. When a dependency
is missing the corresponding public symbols resolve to ``None`` so callers'
``is None`` guard fires.
"""

from agilerl import HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES

if HAS_LLM_DEPENDENCIES:
    from agilerl.algorithms.core.llm_ops.fused_lora import (
        patch_lora_for_fused_forward,
        set_fused_adapter_routing,
        unpatch_lora_for_fused_forward,
        unset_fused_adapter_routing,
    )
else:  # pragma: no cover - LLM deps always present in CI; missing-dep fallback
    patch_lora_for_fused_forward = None
    set_fused_adapter_routing = None
    unpatch_lora_for_fused_forward = None
    unset_fused_adapter_routing = None

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
    # liger-kernel is an optional, Linux-only dependency; keep the names
    # resolvable as None sentinels (callers gate on ``HAS_LIGER_KERNEL``).
    # Mypy-style ``type: ignore`` is used deliberately: the invalid-assignment
    # only surfaces where liger resolves (Linux), so a ``ty: ignore`` would read
    # as unused on macOS/Windows checkouts.
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
