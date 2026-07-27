"""LLM-specific fused-kernel ops (Liger and LoRA).

These modules import optional, Linux-only dependencies (PEFT, liger-kernel), so
they are imported from directly rather than re-exported here: a missing
dependency then raises at the import site instead of yielding a name that is
``None`` and fails somewhere later.
"""
