# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Whether a gradient forward packs its sequences."""

from __future__ import annotations

# Attention backends that build a padding-free (packed) forward. A dense backend
# builds an O(N^2) block-diagonal mask over the packed row instead, so packing
# under one is silently a padded forward.
PACKING_ATTN_IMPLEMENTATIONS = frozenset({"flash_attention_2", "flex_attention"})
# Model types whose recurrent mixers reset their scan and conv state at every
# packed document boundary.
BOUNDARY_RESET_MODEL_TYPES = frozenset(
    {
        "granitemoehybrid",
        "nemotron_h",
        "nemotron_h_omni",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)


def leaks_recurrent_state(model_type: str | None, has_recurrent_layers: bool) -> bool:
    """Whether packing would carry recurrent state across documents of this model.

    :param model_type: Hugging Face ``model_type`` of the checkpoint.
    :param has_recurrent_layers: The decoder has Mamba or linear-attention layers.
    :return: ``True`` when it has recurrent layers that do not reset at boundaries.
    """
    return has_recurrent_layers and model_type not in BOUNDARY_RESET_MODEL_TYPES


def sequence_packing_blocker(
    supports_sequence_packing: bool,
    attn_implementation: str | None,
    row_copies: int,
    recurrent_state_leaks: bool,
    vision_language: bool,
) -> str | None:
    """Why the gradient forward cannot pack its sequences, or ``None`` when it packs.

    :param supports_sequence_packing: The algorithm builds a packed gradient forward.
    :param attn_implementation: The language model's attention backend.
    :param row_copies: Times the packed row is stacked, one per routed
        adapter. LLMPPO's fused actor and critic pass stacks 2.
    :param recurrent_state_leaks: A recurrent mixer would carry state across
        packed documents, as :func:`leaks_recurrent_state`.
    :param vision_language: The model has a vision tower.
    :return: A short reason, or ``None`` when packing applies.
    """
    if not supports_sequence_packing:
        return "this algorithm's gradient forward is padded"
    if attn_implementation not in PACKING_ATTN_IMPLEMENTATIONS:
        return (
            f"attn_implementation {attn_implementation!r} has no varlen or "
            "block-sparse path"
        )
    # Text forwards hand FlashAttention-2 offsets spanning every stacked row.
    # A vision-language forward leaves its text layers to derive them from
    # position_ids, which FlashAttention-2 does for a single row only.
    if (
        attn_implementation == "flash_attention_2"
        and row_copies > 1
        and vision_language
    ):
        return (
            "flash_attention_2 derives a vision-language forward's packed "
            f"offsets from one row; this forward stacks {row_copies}"
        )
    if recurrent_state_leaks:
        return "recurrent mixers do not reset at packed document boundaries"
    return None
