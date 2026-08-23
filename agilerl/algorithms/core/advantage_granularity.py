# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared batch probe for LLM ``advantage_granularity="auto"``."""

from __future__ import annotations

import torch


def batch_is_single_turn(turn_ids: torch.Tensor | None) -> bool:
    """True when every sample has at most one turn, or ``turn_ids`` is missing.

    :param turn_ids: Per-token turn index ``[batch, seq_len]``; ``-1`` for
        padding. ``None`` is treated as a single-turn batch.
    :type turn_ids: torch.Tensor | None
    :return: Whether the batch has no multi-turn structure.
    :rtype: bool
    """
    if turn_ids is None:
        return True
    per_sample_num_turns = turn_ids.max(dim=1).values + 1
    return bool((per_sample_num_turns <= 1).all())


def resolve_batch_advantage_granularity(
    configured: str,
    turn_ids: torch.Tensor | None,
    *,
    single_turn: str,
    multi_turn: str,
    can_use_multi_turn: bool = True,
) -> str:
    """Resolve ``"auto"`` from batch turn structure; pass explicit values through.

    PPO and REINFORCE map single-turn batches to ``"token"`` and multi-turn
    batches to ``"turn"``. GRPO has no token-level advantage, so it maps
    single-turn batches to ``"trajectory"`` and multi-turn batches to
    ``"turn"``.

    :param configured: User setting, including ``"auto"``.
    :type configured: str
    :param turn_ids: Per-token turn indices, or ``None``.
    :type turn_ids: torch.Tensor | None
    :param single_turn: Grain for a single-turn batch.
    :type single_turn: str
    :param multi_turn: Grain when any sample has more than one turn.
    :type multi_turn: str
    :param can_use_multi_turn: False when the batch cannot support ``multi_turn``
        (GRPO: no per-turn rewards).
    :type can_use_multi_turn: bool
    :return: The effective advantage granularity.
    :rtype: str
    """
    if configured != "auto":
        return configured
    if can_use_multi_turn and not batch_is_single_turn(turn_ids):
        return multi_turn
    return single_turn
