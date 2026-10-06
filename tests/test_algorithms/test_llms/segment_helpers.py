# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Segmented-episode batches shared by the LLM ``learn`` segment tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from agilerl.algorithms.core import LLMAlgorithm
from agilerl.components.llm_rollout_data import EpisodeSegments

SEQ_LEN = 10
EPISODE_SEGMENTS = [
    EpisodeSegments(token_lengths=torch.tensor([4, 6])),
    None,
    EpisodeSegments(token_lengths=torch.tensor([5, 3])),
    None,
]
REAL_LENGTHS = [10, 8, 8, 10]
NUM_SEGMENT_ROWS = 6
# 24 action tokens over the 6 segment rows.
MEAN_SEGMENT_ROW_ACTION_TOKENS = 4.0


def episode_batch(pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Four ``T = 10`` episodes laid out per ``EPISODE_SEGMENTS``.

    Each segment's first token is prompt and the position predicting the next
    segment's first token is not an action.
    """
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, pad_token_id, (4, SEQ_LEN), generator=generator)
    mask = torch.zeros(4, SEQ_LEN - 1, dtype=torch.bool)
    for episode, segments in enumerate(EPISODE_SEGMENTS):
        lengths = (
            [REAL_LENGTHS[episode]]
            if segments is None
            else segments.token_lengths.tolist()
        )
        start = 0
        for length in lengths:
            mask[episode, start + 1 : start + length - 1] = True
            start += length
        ids[episode, start:] = pad_token_id
    return ids, mask


def segment_experiences(
    pad_token_id: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """``learn`` experiences of the ``EPISODE_SEGMENTS`` episodes."""
    ids, mask = episode_batch(pad_token_id)
    return list(ids.split(1)), list(mask.split(1)), torch.tensor([1.0, 0.0, 0.0, 1.0])


def episode_sampling_logps(pad_token_id: int) -> list[torch.Tensor]:
    """One flat vLLM sampling log-prob per action token of each episode."""
    _ids, mask = episode_batch(pad_token_id)
    generator = torch.Generator().manual_seed(5)
    return [
        -4.0 + 0.1 * torch.randn(int(row.sum()), generator=generator) for row in mask
    ]


def action_token_policy_loss(
    policy_hidden: torch.Tensor,
    _head_w: torch.Tensor,
    _head_b: torch.Tensor | None,
    _target_ids: torch.Tensor,
    mask: torch.Tensor,
    *_args: Any,
    **_kwargs: Any,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Liger fused policy loss stand-in: zero loss, ``kl`` = the micro-batch's action tokens."""
    zero = torch.tensor(0.0)
    return (policy_hidden * 0.0).sum(), (mask.sum().float(), zero, zero, zero)


def use_fake_liger_policy_loss(
    agent: LLMAlgorithm, monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    """Train ``agent`` on the Liger path with :func:`action_token_policy_loss` as its kernel."""
    monkeypatch.setattr(f"{module}.HAS_LIGER_KERNEL", True)
    monkeypatch.setattr(f"{module}.apply_fused_policy_loss", action_token_policy_loss)
    agent.use_liger_loss = True


def pad_to_eight_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as one of two ranks whose other rank has 8 segment rows.

    The other rank's optimizer steps hold as many real micro-batches as this
    rank's.
    """
    for module in ("agilerl.algorithms.core.base", "agilerl.algorithms.grpo"):
        monkeypatch.setattr(f"{module}.get_world_size", lambda: 2)
        monkeypatch.setattr(
            f"{module}.allreduce_minmax_int",
            lambda value: (value, 8) if value == NUM_SEGMENT_ROWS else (value, value),
        )
    monkeypatch.setattr(
        "agilerl.algorithms.core.base.allreduce_sum_ints",
        lambda values: [2 * int(value) for value in values],
    )


def record_step_gradients(
    agent: LLMAlgorithm, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, torch.Tensor]]:
    """LoRA gradients the optimizer sees at each step of ``agent``."""
    recorded: list[dict[str, torch.Tensor]] = []
    optimizer_step = agent.optimizer.step

    def recording_step(*args: Any, **kwargs: Any) -> None:
        recorded.append(
            {
                name: param.grad.detach().clone()
                for name, param in agent.actor.named_parameters()
                if "lora_" in name and param.grad is not None
            }
        )
        optimizer_step(*args, **kwargs)

    monkeypatch.setattr(agent.optimizer, "step", recording_step)
    return recorded


def lora_weights(agent: LLMAlgorithm) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in agent.actor.named_parameters()
        if "lora_" in name
    }
