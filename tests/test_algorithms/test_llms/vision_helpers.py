# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Vision-language episodes and a tiny VL model shared by the LLM ``learn`` vision tests.

The model adds one vision row to each image placeholder embedding, in row
order, and raises when the counts differ, as a VL forward does.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from agilerl.algorithms.core import LLMAlgorithm
from agilerl.algorithms.core.llm_ops.fused_lora import ROUTING_STATE
from agilerl.components.llm_rollout_data import EpisodeSegments
from tests.test_algorithms.test_llms.llm_helpers import (
    DummyConfig,
    DummyHiddenStatesModel,
)

PAD_TOKEN_ID = 63
IMAGE_TOKEN_ID = 62
VOCAB = 64
PIXEL_DIM = 4
SEQ_LEN = 10
# Per episode: segment token lengths, and the image placeholder positions of
# each segment (relative to the segment start). Episodes 1 and 3 are unsegmented.
EPISODES = [
    ([4, 6], [[0], [0]]),
    ([8], [[0, 1]]),
    ([5, 3], [[0, 1], [0]]),
    ([10], [[0]]),
]
SEGMENTS = [
    EpisodeSegments(
        token_lengths=torch.tensor(lengths),
        pixel_rows=torch.tensor([len(images) for images in placeholders]),
    )
    if len(lengths) > 1
    else None
    for lengths, placeholders in EPISODES
]
IMAGE_COUNTS = [
    sum(len(images) for images in placeholders) for _, placeholders in EPISODES
]


class VisionHiddenStatesModel(DummyHiddenStatesModel):
    """Dummy VL causal LM that records the vision rows and adapter of each row of each forward."""

    def __init__(self, config: DummyConfig) -> None:
        super().__init__(config=config, device="cpu")
        self.vision_proj = nn.Linear(PIXEL_DIM, 32, dtype=self.datatype)
        self.forwards: list[dict[str, Any]] = []

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *args: Any,
        pixel_values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        placeholders = input_ids == IMAGE_TOKEN_ID
        vision_rows = 0 if pixel_values is None else int(pixel_values.shape[0])
        if int(placeholders.sum()) != vision_rows:
            msg = (
                f"{int(placeholders.sum())} image placeholders for "
                f"{vision_rows} vision rows"
            )
            raise ValueError(msg)
        embeds = self.embed(input_ids.long())
        if pixel_values is not None:
            image_embeds = torch.zeros_like(embeds)
            image_embeds[placeholders] = self.vision_proj(pixel_values)
            embeds = embeds + image_embeds
            routing = ROUTING_STATE.get(self.linear_1) or [
                self.linear_1.active_adapters[0]
            ] * int(input_ids.shape[0])
            self.forwards.append(
                {
                    "input_ids": input_ids.detach().clone(),
                    "pixel_values": pixel_values.detach().clone(),
                    "routing": list(routing),
                }
            )
        hidden = self.linear_1(embeds)
        return CausalLMOutputWithPast(
            logits=self.lm_head(hidden), hidden_states=(hidden,)
        )


def vision_config() -> DummyConfig:
    return DummyConfig(input_size=6, max_tokens=4, vocab_size=VOCAB, hidden_size=32)


def vision_episodes() -> tuple[
    list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Token ids, action masks, per-turn rewards, turn ids and vision rows of ``EPISODES``.

    Each segment is one turn. Its first two tokens are prompt, and the position
    predicting the next segment's first token is not an action. An
    unsegmented episode splits its action tokens over two turns. Vision row
    ``k`` is filled with ``k``.
    """
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(
        0, IMAGE_TOKEN_ID, (len(EPISODES), SEQ_LEN), generator=generator
    )
    mask = torch.zeros(len(EPISODES), SEQ_LEN - 1, dtype=torch.bool)
    turn_ids = torch.full((len(EPISODES), SEQ_LEN - 1), -1, dtype=torch.long)
    for episode, (lengths, placeholders) in enumerate(EPISODES):
        start = 0
        for turn, (length, images) in enumerate(
            zip(lengths, placeholders, strict=True)
        ):
            ids[episode, [start + offset for offset in images]] = IMAGE_TOKEN_ID
            mask[episode, start + 1 : start + length - 1] = True
            turn_ids[episode, start + 1 : start + length - 1] = turn
            start += length
        ids[episode, start:] = PAD_TOKEN_ID
        if len(lengths) == 1:
            actions = torch.nonzero(mask[episode]).flatten()
            turn_ids[episode, actions[len(actions) // 2 :]] = 1
    rewards = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    num_vision_rows = sum(IMAGE_COUNTS)
    pixel_values = (
        torch.arange(num_vision_rows, dtype=torch.float32)
        .unsqueeze(-1)
        .expand(num_vision_rows, PIXEL_DIM)
        .contiguous()
    )
    return list(ids.split(1)), list(mask.split(1)), rewards, turn_ids, pixel_values


def expected_vision_rows(segmented: bool) -> dict[tuple[int, ...], list[int]]:
    """Vision row ids each training row's real tokens own, keyed by those tokens."""
    token_ids, *_ = vision_episodes()
    expected: dict[tuple[int, ...], list[int]] = {}
    next_row = 0
    for ids, (lengths, placeholders) in zip(token_ids, EPISODES, strict=True):
        spans = lengths if segmented else [sum(lengths)]
        counts = (
            [len(images) for images in placeholders]
            if segmented
            else [sum(len(images) for images in placeholders)]
        )
        start = 0
        for length, count in zip(spans, counts, strict=True):
            key = tuple(ids[0, start : start + length].tolist())
            expected[key] = list(range(next_row, next_row + count))
            next_row += count
            start += length
    return expected


def routed_rows(
    forwards: list[dict[str, Any]], expected: dict[tuple[int, ...], list[int]]
) -> set[tuple[str, tuple[int, ...]]]:
    """Check each forwarded row got the vision rows ``expected`` gives its real tokens.

    :return: The ``(adapter, real tokens)`` of every forwarded row.
    """
    routed = set()
    for forward in forwards:
        offset = 0
        pixel_ids = forward["pixel_values"][:, 0].long().tolist()
        for row, adapter in zip(forward["input_ids"], forward["routing"], strict=True):
            count = int((row == IMAGE_TOKEN_ID).sum())
            tokens = tuple(row[row != PAD_TOKEN_ID].tolist())
            assert pixel_ids[offset : offset + count] == expected[tokens], (
                adapter,
                tokens,
            )
            offset += count
            routed.add((adapter, tokens))
    return routed


def vision_model(agent: LLMAlgorithm) -> VisionHiddenStatesModel:
    return next(
        module
        for module in agent.actor.modules()
        if isinstance(module, VisionHiddenStatesModel)
    )


def adapter_weights(agent: LLMAlgorithm, adapter: str) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in agent.actor.named_parameters()
        if "lora_" in name and f".{adapter}." in name
    }


def learn_on_vision_episodes(
    agent: LLMAlgorithm,
    *,
    segmented: bool,
    pixel_offset: float = 0.0,
) -> dict[str, float]:
    """Run one ``learn`` on ``EPISODES``, with every vision row shifted by ``pixel_offset``."""
    token_ids, action_masks, rewards, turn_ids, pixel_values = vision_episodes()
    return agent.learn(
        (token_ids, action_masks, rewards),
        turn_ids=turn_ids,
        episode_segments=SEGMENTS if segmented else None,
        pixel_values=pixel_values + pixel_offset,
        pixel_image_counts=IMAGE_COUNTS,
        image_token_id=IMAGE_TOKEN_ID,
    )
