"""Compatibility shims required before importing ``ring_flash_attn``.

``ring-flash-attn`` 0.1.8 imports ``is_flash_attn_greater_or_equal_2_10`` from
``transformers.modeling_flash_attention_utils``, which transformers >= 5.4
removed. Patch the symbol before any ``ring_flash_attn`` import.
"""

from __future__ import annotations


def ensure_ring_flash_attn_transformers_compat() -> None:
    """Install missing transformers helpers that ``ring_flash_attn`` imports."""
    import transformers.modeling_flash_attention_utils as mfau

    if not hasattr(mfau, "is_flash_attn_greater_or_equal_2_10"):
        # FA2 >= 2.10 is the only build AgileRL CP supports; treat as present.
        mfau.is_flash_attn_greater_or_equal_2_10 = lambda: True


# Apply on import so ``from agilerl.utils.ring_attn_compat import …`` is enough
# when callers import this module before ``ring_flash_attn``.
ensure_ring_flash_attn_transformers_compat()
