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
    "mimo_v2": "flex_attention",
    "deepseek_v4": "flex_attention",
    # Hugging Face's GLM-5 Next classes support no other backend.
    "glm5_next": "sdpa",
    "glm5_next_text": "sdpa",
}
# Backends the context-parallel (Ulysses) attention patches.
CONTEXT_PARALLEL_ATTN_IMPLEMENTATIONS = frozenset(
    {"flash_attention_2", "flex_attention"}
)
# Model types whose own context-parallel plan gathers the keys for each
# sequence shard, so context parallel runs on their pinned backend.
SEQUENCE_SHARD_ATTENTION_MODEL_TYPES = frozenset({"glm5_next", "glm5_next_text"})
# Model types checked to train correctly with flash_attention_2 on CUDA and
# sdpa elsewhere. Every catalog model type is here or in the table above.
KERNEL_RULE_MODEL_TYPES = frozenset(
    {
        "granitemoe",
        "granitemoehybrid",
        "llama",
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


def context_parallel_blocker(
    attn_implementation: str, cp: int, model_type: str | None = None
) -> str | None:
    """Why context parallel cannot run on this model and attention backend, or ``None`` when it can.

    :param attn_implementation: The language model's attention backend.
    :param cp: Context-parallel degree.
    :param model_type: Hugging Face ``model_type`` of the language model, or None.
    :return: A short reason, or ``None`` when ``cp`` is 1 or the model or
        backend has a context-parallel path.
    """
    if (
        cp == 1
        or attn_implementation in CONTEXT_PARALLEL_ATTN_IMPLEMENTATIONS
        or model_type in SEQUENCE_SHARD_ATTENTION_MODEL_TYPES
    ):
        return None
    return (
        f"fsdp.cp={cp} needs flash_attention_2 or flex_attention attention; "
        f"this model trains with {attn_implementation}"
    )
