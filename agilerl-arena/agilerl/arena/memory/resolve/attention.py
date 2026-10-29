# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Which attention backend the trainer loads a model with."""

from __future__ import annotations

# Model types whose trainer attention backend is fixed whatever the device.
FAMILY_ATTN_IMPLEMENTATIONS: dict[str, str] = {
    "nemotron_h": "flash_attention_2",
    "nemotron_h_omni": "flash_attention_2",
    "gemma3": "flex_attention",
    "gemma3_text": "flex_attention",
    "gemma4": "flex_attention",
    "gemma4_text": "flex_attention",
    "gpt_oss": "flex_attention",
}
# Model types checked to train correctly with flash_attention_2 on CUDA and
# sdpa elsewhere. Every catalog model type is here or in the table above.
KERNEL_RULE_MODEL_TYPES = frozenset(
    {
        "granitemoe",
        "granitemoehybrid",
        "qwen2",
        "qwen3",
        "qwen3_5",
        "qwen3_5_moe",
    }
)


def resolve_attn_implementation(model_type: str | None, cuda: bool) -> str:
    """Attention backend the trainer loads the language model with.

    :param model_type: Hugging Face ``model_type`` of the checkpoint.
    :param cuda: The trainer runs on a CUDA device.
    :return: The family's backend, else ``flash_attention_2`` on CUDA, else
        ``sdpa``.
    """
    family = FAMILY_ATTN_IMPLEMENTATIONS.get(model_type or "")
    if family is not None:
        return family
    if cuda:
        return "flash_attention_2"
    return "sdpa"
