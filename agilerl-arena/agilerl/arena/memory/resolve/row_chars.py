# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""A dataset's characters per trained row, as a model family's tokens."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from agilerl.arena.memory.resolve.split import RowTokens


class RowChars(NamedTuple):
    """Characters per trained row of an SFT or DPO dataset, before any chat template.

    Characters are Unicode scalar values. An SFT row is prompt + target, or for
    a ``messages`` row, each message's ``content``, its reasoning
    (``reasoning_content``, ``thinking``, ``reasoning``) and the JSON text of
    its ``tool_calls``, plus the JSON text of the row's ``tools``. A DPO pair
    is two rows: prompt + chosen and prompt + rejected.

    :ivar mean: Mean characters per row.
    :ivar mean_square: Mean of each row's squared character count.
    """

    mean: float
    mean_square: float


class TokenizerFit(NamedTuple):
    """``tokens = chars / chars_per_token + template_tokens`` for one family.

    :ivar chars_per_token: Characters one token covers.
    :ivar template_tokens: Tokens the chat template adds to a row.
    """

    chars_per_token: float
    template_tokens: float


# Fitted per Hugging Face model_type with each family's tokenizer and chat
# template on 300-row slices of public SFT and DPO datasets (UltraChat 200k,
# No Robots, GSM8K, CodeAlpaca-20k, UltraFeedback binarized, Orca DPO pairs),
# each dataset weighted the same: mean tokens within 21% on every dataset,
# under on math (GSM8K) and over on code (CodeAlpaca).
TOKENIZER_FITS: Mapping[str, TokenizerFit] = {
    "deepseek_v4": TokenizerFit(5.21, 123.6),
    "gemma4": TokenizerFit(4.92, 61.7),
    "glm5_next": TokenizerFit(4.92, 47.7),
    "gpt_oss": TokenizerFit(5.21, 108.2),
    "granitemoe": TokenizerFit(4.18, 98.5),
    "granitemoehybrid": TokenizerFit(4.93, 64.6),
    "mimo_v2": TokenizerFit(4.86, 50.4),
    "nemotron_h": TokenizerFit(4.81, 60.2),
    "nemotron_h_omni": TokenizerFit(4.81, 60.2),
    "qwen2": TokenizerFit(4.95, 72.2),
    "qwen3": TokenizerFit(4.88, 53.7),
    "qwen3_5": TokenizerFit(4.87, 58.2),
    "qwen3_5_moe": TokenizerFit(4.87, 59.3),
}
# Families outside the table: the median of the fits.
DEFAULT_TOKENIZER_FIT = TokenizerFit(4.88, 60.2)


def row_tokens_from_chars(row_chars: RowChars, model_type: str) -> RowTokens:
    """A dataset's row lengths in a model family's tokens.

    Each row gains the family's template tokens ``o`` and covers ``r``
    characters per token, so the mean is ``mean / r + o`` and the mean square
    ``mean_square / r**2 + 2 * o * mean / r + o**2``.

    :param row_chars: The dataset's characters per trained row.
    :param model_type: Hugging Face ``model_type`` of the trained model; one
        outside :data:`TOKENIZER_FITS` takes :data:`DEFAULT_TOKENIZER_FIT`.
    :return: The dataset's tokens per trained row.
    :raises ValueError: If a character stat is negative.
    """
    if row_chars.mean < 0 or row_chars.mean_square < 0:
        msg = f"Row character stats must not be negative: {row_chars}."
        raise ValueError(msg)
    ratio, overhead = TOKENIZER_FITS.get(model_type, DEFAULT_TOKENIZER_FIT)
    mean = row_chars.mean / ratio
    return RowTokens(
        mean=mean + overhead,
        mean_square=row_chars.mean_square / ratio**2
        + 2 * overhead * mean
        + overhead**2,
    )
