# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Token ids vLLM could sample at each generated token, read from its completions."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm import CompletionOutput


def extract_vllm_sampling_supports(output: CompletionOutput) -> torch.Tensor:
    """Read the token ids vLLM could sample at each token of a ``CompletionOutput``.

    :param output: Completion of a request to an engine built with
        ``return_sampling_mask=True``.
    :type output: CompletionOutput
    :return: ``(n, K)`` int32 support per generated token, ``-1`` padded.
    :rtype: torch.Tensor
    :raises RuntimeError: If the output has no sampling mask, or one that does
        not cover every generated token.
    """
    mask = output.sampling_mask
    n_tokens = len(output.token_ids)
    if mask is None or len(mask.token_ids) != n_tokens:
        covered = None if mask is None else len(mask.token_ids)
        msg = (
            f"vLLM returned sampling supports for {covered} of {n_tokens} generated "
            "tokens; return_sampling_mask needs vLLM's V2 model runner, which "
            "records one support per sampled token"
        )
        raise RuntimeError(msg)
    lengths = np.fromiter(map(len, mask.token_ids), dtype=np.int64, count=n_tokens)
    supports = np.full((n_tokens, max(int(lengths.max(initial=0)), 1)), -1, np.int32)
    starts = np.cumsum(lengths) - lengths
    flat = np.fromiter(chain.from_iterable(mask.token_ids), dtype=np.int32)
    rows = np.repeat(np.arange(n_tokens), lengths)
    supports[rows, np.arange(flat.size) - np.repeat(starts, lengths)] = flat
    return torch.from_numpy(supports)


def concat_sampling_supports(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    """Stack per-token supports of consecutive generations, ``-1`` padding the width.

    :param parts: ``(n_i, K_i)`` supports in generation order.
    :type parts: Sequence[torch.Tensor]
    :return: ``(sum n_i, max K_i)`` supports.
    :rtype: torch.Tensor
    """
    if not parts:
        return torch.empty(0, 1, dtype=torch.int32)
    width = max(int(part.shape[1]) for part in parts)
    return torch.cat(
        [
            torch.nn.functional.pad(part, (0, width - int(part.shape[1])), value=-1)
            for part in parts
        ]
    )
