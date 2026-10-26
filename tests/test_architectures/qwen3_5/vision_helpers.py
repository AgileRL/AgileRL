# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tiny Qwen3.5 VL model and image prompts for vision-row tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)

from agilerl.algorithms.core.llm_ops.vision_lockstep import FILLER_METHOD
from agilerl.architectures.qwen3_5.vision import ACCEPTS_VISION_ROWS
from tests.test_architectures.qwen3_5.gdn_helpers import restored_qwen_gdn_classes

IMAGE, START, END = 100, 101, 102
GRID = 4
TOKENS_PER_IMAGE = GRID * GRID // 4
PATCH_DIM = 3 * 2 * 2 * 2


@contextmanager
def restored_qwen_vl_model() -> Iterator[None]:
    """Undo the class-level GDN and vision-row patches of the Qwen3.5 VL installer on exit."""
    forward = Qwen3_5Model.__dict__["forward"]
    try:
        with restored_qwen_gdn_classes():
            yield
    finally:
        Qwen3_5Model.forward = forward
        for name in (ACCEPTS_VISION_ROWS, FILLER_METHOD):
            if name in vars(Qwen3_5Model):
                delattr(Qwen3_5Model, name)


def tiny_qwen_vl() -> Qwen3_5ForConditionalGeneration:
    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "intermediate_size": 64,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "full_attention_interval": 2,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 2,
        },
        image_token_id=IMAGE,
        vision_start_token_id=START,
        vision_end_token_id=END,
        video_token_id=103,
    )
    return Qwen3_5ForConditionalGeneration(config).eval()


def image_prompt(prefix: list[int], images: int) -> list[int]:
    ids = list(prefix)
    for _ in range(images):
        ids += [START, *([IMAGE] * TOKENS_PER_IMAGE), END, 7]
    return ids


def flat_inputs(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grids = torch.tensor([[1, GRID, GRID]] * rows.shape[0])
    return rows.reshape(-1, PATCH_DIM), grids
