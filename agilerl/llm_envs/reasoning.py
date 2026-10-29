# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Token-id helpers for a turn's reasoning block."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

REASONING_END_MARKER = "</think>"
"""Marker closing a reasoning block in the tag format, or taught as plain text by a prompt."""

SUBSEQUENCE_NOT_FOUND = -1
"""Index reported when a token-id pattern does not occur in a token-id sequence."""


def tokenizer_reasoning_end_ids(
    tokenizer: PreTrainedTokenizerBase, marker: str
) -> tuple[int, ...]:
    """Token ids of ``marker``: its vocab id, else its encoding without special tokens.

    The vocab holds the marker whenever vLLM can force it through the sampler at
    ``thinking_token_budget``; the encoding covers a marker spelled as several tokens.

    :param tokenizer: Tokenizer the rollout generates with.
    :type tokenizer: PreTrainedTokenizerBase
    :param marker: Reasoning-end marker text, such as ``</think>``.
    :type marker: str
    :return: Marker token ids.
    :rtype: tuple[int, ...]
    """
    marker_id = tokenizer.convert_tokens_to_ids(marker)
    # Hugging Face maps a token missing from the vocab to the unk ID, or to
    # None when the tokenizer has no unk token (gpt-oss).
    if isinstance(marker_id, int) and marker_id != tokenizer.unk_token_id:
        return (marker_id,)
    return tuple(
        int(token_id) for token_id in tokenizer.encode(marker, add_special_tokens=False)
    )


def find_token_subsequence(
    token_ids: Sequence[int],
    pattern: Sequence[int],
) -> int:
    """Locate the first occurrence of a token-id pattern in a token-id sequence.

    :param token_ids: Sequence to search.
    :type token_ids: Sequence[int]
    :param pattern: Token ids to locate; an empty pattern never matches.
    :type pattern: Sequence[int]
    :return: Index the pattern starts at, or ``SUBSEQUENCE_NOT_FOUND``.
    :rtype: int
    """
    pattern_len = len(pattern)
    if pattern_len == 0 or pattern_len > len(token_ids):
        return SUBSEQUENCE_NOT_FOUND
    head = pattern[0]
    last_start = len(token_ids) - pattern_len
    start = 0
    while start <= last_start:
        try:
            index = token_ids.index(head, start, last_start + 1)
        except ValueError:
            return SUBSEQUENCE_NOT_FOUND
        if all(
            token_ids[index + offset] == pattern[offset]
            for offset in range(1, pattern_len)
        ):
            return index
        start = index + 1
    return SUBSEQUENCE_NOT_FOUND


def reasoning_trace_text(
    gen_ids: Sequence[int],
    special_ids: Collection[int],
    decode: Callable[..., str],
) -> str:
    """Decoded turn with its reasoning markers, trailing end-of-turn tokens dropped.

    Special tokens inside the turn stay, so Gemma 4 channel markers and gpt-oss
    harmony channel headers reach a format reward as the model wrote them.

    :param gen_ids: Token ids generated for the turn, prompt excluded.
    :type gen_ids: Sequence[int]
    :param special_ids: The tokenizer's special-token ids.
    :type special_ids: Collection[int]
    :param decode: Tokenizer-style decode taking ``skip_special_tokens``.
    :type decode: Callable[..., str]
    :return: The turn's text, stripped.
    :rtype: str
    """
    ids = list(gen_ids)
    while ids and ids[-1] in special_ids:
        ids.pop()
    return decode(ids, skip_special_tokens=False).strip()
