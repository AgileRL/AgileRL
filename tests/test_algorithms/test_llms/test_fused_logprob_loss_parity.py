# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Parity of every LLM algorithm loss on fused log-probs against an unfused reference.

Each loss runs once on :class:`FusedLinearLogProbsFunction` log-probs and once
on a full-vocabulary fp32 ``log_softmax`` over the same bf16 hidden states and
head. The fused log-probs are fp32; losses and hidden/head gradients match the
reference.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from peft import LoraConfig

from agilerl.algorithms.cispo import CISPO
from agilerl.algorithms.core.llm_ops.fused_logprobs import (
    FusedLinearLogProbsFunction,
)
from agilerl.algorithms.dpo import DPO
from agilerl.algorithms.grpo import GRPO
from agilerl.algorithms.gspo import GSPO
from agilerl.algorithms.ppo_llm import PPO as LLMPPO
from agilerl.utils.llm_utils import clipped_is_surrogate
from tests.test_algorithms.test_llms.llm_helpers import (
    create_module,
    create_value_head_module,
)

ROWS, SEQ, WIDTH, VOCAB = 4, 7, 16, 40
# Uneven chunks: 28 flat rows over chunks of 5.
CHUNK_ROWS = 5
PAD_TOKEN_ID = VOCAB - 1
LEVELS = ("token", "turn", "trajectory")
# Both paths compute in fp32 from the same bf16 operands; only summation order
# differs.
LOGPROB_TOL = {"atol": 1e-5, "rtol": 0.0}
LOSS_TOL = {"atol": 1e-5, "rtol": 1e-5}
# Gradients land in the bf16 operands' dtype (torch's bf16 tolerance). The
# fused head gradient sums per-chunk bf16 pieces, about one bf16 ulp off.
HIDDEN_GRAD_TOL = {"atol": 1e-5, "rtol": 1.6e-2}
HEAD_GRAD_TOL = {"atol": 1e-3, "rtol": 1.6e-2}

LossFn = Callable[[torch.Tensor], torch.Tensor]


def head_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 ``(B, T, H)`` hidden states, bf16 ``(V, H)`` head and ``(B, T)`` targets."""
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(ROWS, SEQ, WIDTH, generator=generator)
    weight = torch.randn(VOCAB, WIDTH, generator=generator) / WIDTH**0.5
    targets = torch.randint(0, VOCAB, (ROWS, SEQ), generator=generator)
    return hidden.to(torch.bfloat16), weight.to(torch.bfloat16), targets


def fused_logprobs(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return FusedLinearLogProbsFunction.apply(
        hidden, weight, None, targets, 1.0, CHUNK_ROWS, None
    )


def reference_logprobs(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    logits = hidden.float() @ weight.float().t()
    return logits.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def rollout_tensors() -> dict[str, torch.Tensor]:
    """Fixed action mask, turn ids, advantages and old/reference log-probs."""
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        policy = reference_logprobs(*head_inputs())
    mask = torch.ones(ROWS, SEQ, dtype=torch.bool)
    mask[:, :2] = False
    mask[1, -2:] = False
    turn_ids = torch.full((ROWS, SEQ), -1, dtype=torch.long)
    turn_ids[:, 2:4] = 0
    turn_ids[:, 4:] = 1
    turn_ids[~mask] = -1
    return {
        "mask": mask,
        "turn_ids": turn_ids,
        "old": policy + 0.3 * torch.randn(ROWS, SEQ, generator=generator),
        "reference": policy + 0.2 * torch.randn(ROWS, SEQ, generator=generator),
        "row_advantages": torch.tensor([[1.0], [-0.5], [0.25], [-1.5]]),
        "token_advantages": torch.randn(ROWS, SEQ, generator=generator),
    }


ROLLOUT = rollout_tensors()


def run_loss(
    logprob_fn: Callable[..., torch.Tensor], loss_fn: LossFn
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Log-probs, loss and the hidden and head gradients of one path."""
    hidden, weight, targets = head_inputs()
    hidden.requires_grad_(True)
    weight.requires_grad_(True)
    logps = logprob_fn(hidden, weight, targets)
    loss = loss_fn(logps)
    loss.backward()
    return logps.detach(), loss.detach(), hidden.grad, weight.grad


def assert_matches_reference(loss_fn: LossFn) -> None:
    fused_logps, fused_loss, fused_hidden_grad, fused_weight_grad = run_loss(
        fused_logprobs, loss_fn
    )
    ref_logps, ref_loss, ref_hidden_grad, ref_weight_grad = run_loss(
        reference_logprobs, loss_fn
    )

    assert fused_logps.dtype == torch.float32
    torch.testing.assert_close(fused_logps, ref_logps, **LOGPROB_TOL)
    assert torch.isfinite(fused_loss)
    torch.testing.assert_close(fused_loss, ref_loss, **LOSS_TOL)
    assert ref_hidden_grad.abs().sum() > 0
    torch.testing.assert_close(fused_hidden_grad, ref_hidden_grad, **HIDDEN_GRAD_TOL)
    torch.testing.assert_close(fused_weight_grad, ref_weight_grad, **HEAD_GRAD_TOL)


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["linear_1"],
        task_type="CAUSAL_LM",
        lora_dropout=0.0,
    )


def make_grpo_family(
    cls: type[GRPO], level: str, kl_advantage_shaping: bool = False
) -> GRPO:
    torch.manual_seed(0)
    return cls(
        actor_network=create_module(
            input_size=6, max_tokens=4, vocab_size=VOCAB, device="cpu"
        ),
        pad_token_id=PAD_TOKEN_ID,
        pad_token="<pad>",
        batch_size=2,
        group_size=2,
        beta=0.04,
        use_kl_advantage_shaping=kl_advantage_shaping,
        importance_sampling_level=level,
        loss_norm="micro_batch",
        max_output_tokens=4,
        max_model_len=12,
        micro_batch_size_per_gpu=2,
        wrap=False,
        calc_position_embeddings=False,
        device="cpu",
        lora_config=lora_config(),
    )


def grpo_family_loss(agent: GRPO) -> LossFn:
    def loss_fn(logps: torch.Tensor) -> torch.Tensor:
        loss, _kl, _clipfrac = agent._loss_fn(
            ROLLOUT["mask"],
            logps,
            ROLLOUT["old"],
            ROLLOUT["reference"],
            ROLLOUT["row_advantages"],
            ROLLOUT["turn_ids"],
        )
        return loss

    return loss_fn


class TestGRPOFamilyLossParity:
    @pytest.mark.parametrize("level", LEVELS)
    @pytest.mark.parametrize("cls", [GRPO, CISPO], ids=["grpo", "cispo"])
    def test_loss_matches_unfused_reference(self, cls: type[GRPO], level: str):
        assert_matches_reference(grpo_family_loss(make_grpo_family(cls, level)))

    def test_gspo_loss_matches_unfused_reference(self):
        assert_matches_reference(grpo_family_loss(make_grpo_family(GSPO, "trajectory")))

    @pytest.mark.parametrize("level", LEVELS)
    def test_kl_advantage_shaping_matches_unfused_reference(self, level: str):
        agent = make_grpo_family(GRPO, level, kl_advantage_shaping=True)

        assert_matches_reference(grpo_family_loss(agent))


class TestPPOPolicyLossParity:
    @pytest.mark.parametrize("level", LEVELS)
    def test_loss_matches_unfused_reference(self, level: str):
        # Arrange
        torch.manual_seed(0)
        agent = LLMPPO(
            actor_network=create_value_head_module(
                input_size=6, max_tokens=4, vocab_size=VOCAB, device="cpu"
            ),
            pad_token_id=PAD_TOKEN_ID,
            pad_token="<pad>",
            batch_size=4,
            beta=0.04,
            importance_sampling_level=level,
            max_output_tokens=4,
            max_model_len=12,
            micro_batch_size_per_gpu=2,
            wrap=False,
            calc_position_embeddings=False,
            device="cpu",
            lora_config=LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["linear_1"],
                task_type="CAUSAL_LM",
                lora_dropout=0.0,
                modules_to_save=["summary"],
            ),
        )

        def loss_fn(logps: torch.Tensor) -> torch.Tensor:
            loss, _metrics = agent._ppo_policy_loss(
                logps,
                ROLLOUT["mask"],
                ROLLOUT["old"],
                ROLLOUT["reference"],
                ROLLOUT["token_advantages"],
                ROLLOUT["turn_ids"],
                2,
                "token",
            )
            return loss

        # Act / Assert
        assert_matches_reference(loss_fn)


class TestReinforceSurrogateParity:
    @pytest.mark.parametrize("turn_reduction", ["mean", "sum"])
    @pytest.mark.parametrize("level", LEVELS)
    def test_loss_matches_unfused_reference(self, level: str, turn_reduction: str):
        def loss_fn(logps: torch.Tensor) -> torch.Tensor:
            loss, _clipfrac = clipped_is_surrogate(
                logps - ROLLOUT["old"],
                ROLLOUT["token_advantages"],
                ROLLOUT["mask"],
                ROLLOUT["turn_ids"],
                level,
                0.2,
                turn_reduction=turn_reduction,
                num_turns=2,
            )
            return loss

        assert_matches_reference(loss_fn)


class TestSFTLossParity:
    def test_loss_matches_unfused_reference(self):
        def loss_fn(logps: torch.Tensor) -> torch.Tensor:
            scored = logps[ROLLOUT["mask"]]
            return -scored.sum() / scored.numel()

        assert_matches_reference(loss_fn)


class TestDPOLossParity:
    @pytest.mark.parametrize("nll_alpha", [0.0, 0.5])
    def test_loss_matches_unfused_reference(self, nll_alpha: float):
        # Arrange
        torch.manual_seed(0)
        agent = DPO(
            actor_network=create_module(
                input_size=10, max_tokens=20, vocab_size=VOCAB, device="cpu"
            ),
            pad_token_id=PAD_TOKEN_ID,
            pad_token="<pad>",
            lora_config=lora_config(),
            batch_size=4,
            micro_batch_size_per_gpu=2,
            nll_alpha=nll_alpha,
            wrap=False,
            device="cpu",
        )
        mask = ROLLOUT["mask"].float()
        reference = ROLLOUT["reference"]

        def loss_fn(logps: torch.Tensor) -> torch.Tensor:
            loss, _chosen_reward, _rejected_reward = agent._dpo_loss_standard(
                logps[:2],
                logps[2:],
                reference[:2],
                reference[2:],
                mask[:2],
                mask[2:],
            )
            return loss

        # Act / Assert
        assert_matches_reference(loss_fn)


class TestLoraAgentFrozenHead:
    def test_gradient_forward_leaves_the_head_without_a_gradient(self):
        # Arrange
        agent = make_grpo_family(GRPO, "token")
        ids = torch.randint(0, PAD_TOKEN_ID, (2, 8))
        lm_head = agent._get_lm_head()

        # Act
        logps = agent._get_logprobs(ids, batch_size=2)
        logps.sum().backward()

        # Assert
        assert logps.dtype == torch.float32
        assert not lm_head.weight.requires_grad
        assert lm_head.weight.grad is None
        assert any(
            param.grad is not None
            for name, param in agent.actor.named_parameters()
            if "lora_" in name
        )
