# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GRPO learn with cp=2 gives the LoRA grads and loss of a cp=1 learn on CPU gloo."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from peft import LoraConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from agilerl.algorithms.core.sequence_layout import (
    ContextParallelSequence,
    PackedContextParallelSequence,
)
from agilerl.algorithms.grpo import GRPO
from agilerl.distributed import FSDPConfig
from tests.parallel_helpers import install_cpu_flash_attn, plain
from tests.rank_launch import run_ranks
from tests.test_algorithms.test_llms.segment_helpers import (
    RESOLVED_FSDP_MEMORY,
    record_step_gradients,
    seed_lora_weights,
)

PAD_TOKEN_ID = 31
SEQ = 8
GROUP = 2
PROMPTS = 2
ROW_LENGTHS = (8, 6, 7, 8)
TRAINABLE_NAME_PARTS = ("lora_", "modules_to_save")


def qwen3_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    return Qwen3ForCausalLM(config)


def experiences() -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Right-padded rows of :data:`ROW_LENGTHS`; the first token of each is the prompt."""
    generator = torch.Generator().manual_seed(1)
    token_ids, masks = [], []
    for length in ROW_LENGTHS:
        ids = torch.randint(0, PAD_TOKEN_ID, (1, SEQ), generator=generator)
        ids[0, length:] = PAD_TOKEN_ID
        mask = torch.zeros(1, SEQ - 1, dtype=torch.bool)
        mask[0, : length - 1] = True
        token_ids.append(ids)
        masks.append(mask)
    return token_ids, masks, torch.tensor([1.0, 0.0, 0.0, 1.0])


def make_grpo(
    model: Qwen3ForCausalLM,
    gradient_checkpointing: bool,
    fsdp_config: FSDPConfig | None,
    modules_to_save: list[str] | None = None,
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
            max_model_len=64,
            micro_batch_size_per_gpu=2,
            mini_batch_size=4,
            lora_config=LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
                modules_to_save=modules_to_save,
            ),
            fsdp_config=fsdp_config,
            gradient_checkpointing=gradient_checkpointing,
            use_separate_reference_adapter=False,
            device="cpu",
        )
    seed_lora_weights(agent)
    return agent


def cp_learn_worker(
    rank: int,
    _world_size: int,
    gradient_checkpointing: bool,
    modules_to_save: list[str] | None = None,
    packed: bool = True,
) -> tuple[float, list[dict[str, np.ndarray]]]:
    """One cp=2 rank: the reported loss and the trainable grads of each optimizer step."""
    # The default FSDP2 mesh follows the host accelerator (MPS on macOS);
    # these ranks train on CPU.
    torch._C._get_accelerator = lambda: torch.device("cpu")
    install_cpu_flash_attn()
    model = qwen3_model()
    model.config._attn_implementation = "flash_attention_2"
    agent = make_grpo(
        model,
        gradient_checkpointing,
        FSDPConfig(
            cp=2, param_dtype="float32", reduce_dtype="float32", **RESOLVED_FSDP_MEMORY
        ),
        modules_to_save,
    )
    assert type(agent.sequence_layout) is PackedContextParallelSequence
    if not packed:
        agent.sequence_layout = ContextParallelSequence(
            PAD_TOKEN_ID, 2, agent.sequence_layout.group, rank, 2
        )
    steps = record_step_gradients(agent, pytest.MonkeyPatch(), TRAINABLE_NAME_PARTS)
    loss = agent.learn(experiences())["loss"]
    # NumPy arrays: a torch tensor sent to the parent shares memory that dies
    # with this process before the parent reads it.
    return loss, [
        {plain(name): grad.numpy() for name, grad in s.items()} for s in steps
    ]


class TestGRPOLearnContextParallel:
    @pytest.mark.multi_rank
    @pytest.mark.parametrize("gradient_checkpointing", [False, True])
    @pytest.mark.parametrize("packed", [True, False])
    def test_cp2_matches_cp1_grads_and_loss(
        self,
        gradient_checkpointing: bool,
        packed: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange: the cp=1 reference runs eager attention on one process.
        reference = make_grpo(qwen3_model(), gradient_checkpointing, None)
        reference_steps = record_step_gradients(
            reference, monkeypatch, TRAINABLE_NAME_PARTS
        )

        # Act
        reference_loss = reference.learn(experiences())["loss"]
        reports = run_ranks(
            cp_learn_worker,
            2,
            gradient_checkpointing,
            None,
            packed,
            backend=None,
            timeout=300,
            cpu=True,
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

    @pytest.mark.multi_rank
    def test_trainable_lm_head_matches_cp1_grads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: each CP rank backprops only its own tokens through the head.
        reference = make_grpo(qwen3_model(), False, None, ["lm_head"])
        reference_steps = record_step_gradients(
            reference, monkeypatch, TRAINABLE_NAME_PARTS
        )

        # Act
        reference_loss = reference.learn(experiences())["loss"]
        reports = run_ranks(
            cp_learn_worker, 2, False, ["lm_head"], backend=None, timeout=300, cpu=True
        )

        # Assert: same fp32 tolerance as the LoRA-only case.
        assert any("lm_head" in name for name in reference_steps[0])
        for loss, steps in reports:
            assert loss == pytest.approx(reference_loss, abs=1e-6, rel=1e-5)
            assert len(steps) == len(reference_steps)
            for step, expected in zip(steps, reference_steps, strict=True):
                assert step.keys() == {plain(name) for name in expected}
                for name, grad in expected.items():
                    torch.testing.assert_close(
                        torch.from_numpy(step[plain(name)]), grad, atol=1e-6, rtol=1e-5
                    )
