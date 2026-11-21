# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GRPO learn on images with cp=2 gives the LoRA grads and loss of a cp=1 learn on CPU gloo."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from peft import LoraConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

from agilerl.algorithms.core.sequence_layout import ContextParallelSequence
from agilerl.algorithms.grpo import GRPO
from agilerl.distributed import FSDPConfig
from tests.parallel_helpers import cpu_flex_attention, plain
from tests.rank_launch import run_ranks
from tests.test_algorithms.test_llms.segment_helpers import (
    RESOLVED_FSDP_MEMORY,
    record_step_gradients,
    seed_lora_weights,
)
from tests.test_architectures.gemma4.test_vision import (
    GRID,
    IMAGE,
    PATCH_DIM,
    TOKENS_PER_IMAGE,
    tiny_gemma4,
)

PAD_TOKEN_ID = 0
# Flex rounds the CP frame to 128 columns, so shard 1 starts at column 64.
SEQ = 72
GROUP = 2
PROMPTS = 2
# Image start column of each row: one across the shard boundary, two in the
# first shard, none, and one in the second shard.
IMAGE_STARTS = ((62,), (1, 10), (), (64,))
# Each row's completion starts after its last image and fills the row.
COMPLETION_STARTS = (66, 14, 20, 68)
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)"


def gemma4_model() -> Gemma4ForConditionalGeneration:
    model = tiny_gemma4()
    model.config.get_text_config()._attn_implementation = "flex_attention"
    return model


def image_experiences() -> tuple[
    tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor],
    torch.Tensor,
    list[int],
]:
    """Rows laid out per :data:`IMAGE_STARTS`, their vision rows, and the image count of each row."""
    generator = torch.Generator().manual_seed(1)
    token_ids, masks = [], []
    for starts, completion in zip(IMAGE_STARTS, COMPLETION_STARTS, strict=True):
        ids = torch.randint(1, IMAGE, (1, SEQ), generator=generator)
        for start in starts:
            ids[0, start : start + TOKENS_PER_IMAGE] = IMAGE
        mask = torch.zeros(1, SEQ - 1, dtype=torch.bool)
        mask[0, completion - 1 :] = True
        token_ids.append(ids)
        masks.append(mask)
    counts = [len(starts) for starts in IMAGE_STARTS]
    pixel_values = torch.randn(sum(counts), GRID, GRID, PATCH_DIM, generator=generator)
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    return (token_ids, masks, rewards), pixel_values, counts


def make_grpo(
    model: Gemma4ForConditionalGeneration, fsdp_config: FSDPConfig | None
) -> GRPO:
    # Liger's Triton kernels need a GPU; these agents run on CPU.
    with patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", False):
        agent = GRPO(
            actor_network=model,
            pad_token_id=PAD_TOKEN_ID,
            pad_token="<pad>",
            beta=0.0,
            lr=1e-2,
            max_grad_norm=None,
            group_size=GROUP,
            batch_size=PROMPTS,
            max_model_len=256,
            micro_batch_size_per_gpu=2,
            mini_batch_size=4,
            lora_config=LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=LORA_TARGETS,
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
            ),
            fsdp_config=fsdp_config,
            use_separate_reference_adapter=False,
            device="cpu",
        )
    seed_lora_weights(agent)
    return agent


def learn(agent: GRPO) -> float:
    experiences, pixel_values, counts = image_experiences()
    return agent.learn(
        experiences,
        pixel_values=pixel_values,
        pixel_image_counts=counts,
        image_token_id=IMAGE,
    )["loss"]


def cp_vision_learn_worker(
    rank: int, _world_size: int
) -> tuple[float, list[dict[str, np.ndarray]]]:
    """One cp=2 rank: the reported loss and the trainable grads of each optimizer step."""
    # The default FSDP2 mesh follows the host accelerator (MPS on macOS);
    # these ranks train on CPU.
    torch._C._get_accelerator = lambda: torch.device("cpu")
    # CPU flex has no backward; the stand-in runs the same mask as dense math.
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = cpu_flex_attention
    agent = make_grpo(
        gemma4_model(),
        FSDPConfig(
            cp=2, param_dtype="float32", reduce_dtype="float32", **RESOLVED_FSDP_MEMORY
        ),
    )
    assert type(agent.sequence_layout) is ContextParallelSequence
    assert agent.sequence_layout.rank == rank
    steps = record_step_gradients(agent, pytest.MonkeyPatch())
    loss = learn(agent)
    # NumPy arrays: a torch tensor sent to the parent shares memory that dies
    # with this process before the parent reads it.
    return loss, [
        {plain(name): grad.numpy() for name, grad in s.items()} for s in steps
    ]


class TestGRPOLearnContextParallelImages:
    @pytest.mark.multi_rank
    def test_cp2_matches_cp1_grads_and_loss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: the cp=1 reference runs the same flex math on one process.
        monkeypatch.setitem(
            ALL_ATTENTION_FUNCTIONS, "flex_attention", cpu_flex_attention
        )
        reference = make_grpo(gemma4_model(), None)
        reference_steps = record_step_gradients(reference, monkeypatch)

        # Act
        reference_loss = learn(reference)
        reports = run_ranks(
            cp_vision_learn_worker, 2, backend=None, timeout=300, cpu=True
        )

        # Assert: fp32 differs from the reference only in the all-to-all and
        # gather reduction order.
        assert reference_steps
        for loss, steps in reports:
            assert loss == pytest.approx(reference_loss, abs=1e-6, rel=1e-5)
            assert len(steps) == len(reference_steps)
            for step, expected in zip(steps, reference_steps, strict=True):
                assert step.keys() == {plain(name) for name in expected}
                for name, grad in expected.items():
                    torch.testing.assert_close(
                        torch.from_numpy(step[plain(name)]), grad, atol=1e-6, rtol=1e-5
                    )
