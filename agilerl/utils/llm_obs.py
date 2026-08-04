# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""LLM observation and prompt-budget helpers, kept free of heavy imports.

Separate from ``llm_utils`` so env workers do not import transformers/peft.
"""

from collections.abc import Mapping
from typing import TypeGuard

import torch

from agilerl.typing import ReasoningPrompts

__all__ = ["is_reasoning_prompts", "max_prompt_tokens_for_sliding_window"]


def max_prompt_tokens_for_sliding_window(
    max_model_len: int,
    max_output_tokens: int | None,
) -> int:
    """Upper bound on prompt tokens so at least one completion token can be generated.

    Reserve generation headroom while keeping prompt budget as large as possible.
    When ``max_output_tokens`` is provided, reserve up to that many tokens
    (capped by ``max_model_len``). When it is ``None``, reserve exactly one
    token so generation remains possible without collapsing prompt budget.

    :param max_model_len: Engine context length (prompt + completion ceiling).
    :type max_model_len: int
    :param max_output_tokens: Configured completion cap; if ``None``, reserve
        one token of generation headroom.
    :type max_output_tokens: int | None
    :return: Largest allowed prompt length under that headroom (may be 0).
    :rtype: int
    """
    gen_reserve = (
        max(1, min(max_output_tokens, max_model_len))
        if max_output_tokens is not None
        else 1
    )
    return max(0, max_model_len - gen_reserve)


def is_reasoning_prompts(obs: Mapping[str, object]) -> TypeGuard[ReasoningPrompts]:
    """Check whether a mapping is a tokenized ``ReasoningPrompts`` observation.

    :param obs: An observation mapping returned by a tokenized multi-turn env.
    :type obs: Mapping[str, object]
    :return: ``True`` when the mapping carries prompt tensors.
    :rtype: TypeGuard[ReasoningPrompts]
    """
    return isinstance(obs.get("input_ids"), torch.Tensor) and isinstance(
        obs.get("attention_mask"), torch.Tensor
    )
