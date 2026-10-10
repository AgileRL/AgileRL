# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Parity of the fused Liger GRPO-family loss with the standard PyTorch loss.

Each case runs one ``learn`` on two identical tiny fp32 agents, one per path,
over rows of unequal action lengths split into four accumulated
micro-batches, and compares the optimizer-step gradients and the learn
metrics. The real-kernel cases need ``liger-kernel`` (Linux only).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl import HAS_LIGER_KERNEL
from agilerl.algorithms import grpo as grpo_module
from agilerl.algorithms.grpo import GRPO
from tests.test_algorithms.test_llms.segment_helpers import (
    EPISODE_SEGMENTS,
    segment_experiences,
)
from tests.test_algorithms.test_llms.test_grpo_episode_loss_norm import KERNEL_NAME
from tests.test_algorithms.test_llms.test_grpo_off_policy_masks import (
    ICEPOP_BAND,
    RecordingFusedKernel,
)
from tests.test_algorithms.test_llms.test_grpo_old_logprobs import (
    PAD_TOKEN_ID,
    PROMPT_LEN,
    _make_grpo,
    _record_step_gradients,
)

SEQ_LEN = 8
ACTION_ENDS = (7, 5, 6, 3, 7, 4, 6, 5)
"""Last action position of each row; the rows hold 5, 3, 4, 1, 5, 2, 4, 3 action tokens."""
NUM_ROWS = len(ACTION_ENDS)
REWARDS = (1.0, 0.0, 0.2, 0.9, 0.0, 0.7, 0.5, 0.1)
SAMPLING_NOISE_STD = 0.4
SAMPLING_ROW_DRIFT = (0.0, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0)
"""Rows 1 and 2 lose their groups; this drift puts them past the sequence mask."""
TURN_REWARDS = (
    (1.0, 0.0),
    (0.0, 0.5),
    (0.2, 1.0),
    (0.9, 0.3),
    (0.0, 0.4),
    (0.7, 0.0),
    (0.5, 0.9),
    (0.1, 0.2),
)
KL_CLAMP = 0.02
VLLM_IS_CAP = 1.2
SEQUENCE_MASK_THRESHOLD = 0.05
LOSS_TYPE_ARG = 14
IMPORTANCE_SAMPLING_LEVEL_ARG = 16
NUM_ITEMS_ARG = 30
# fp32 throughout; Liger's chunked log-softmax and reductions sum in a
# different order from the eager path, which moves results by ~1e-7.
LOSS_RTOL = 1e-5
GRAD_RTOL = 1e-4
ATOL = 1e-6


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids and action masks of rows that end at ``ACTION_ENDS``, then pad."""
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, PAD_TOKEN_ID, (NUM_ROWS, SEQ_LEN), generator=generator)
    mask = torch.zeros(NUM_ROWS, SEQ_LEN - 1, dtype=torch.bool)
    for row, end in enumerate(ACTION_ENDS):
        mask[row, PROMPT_LEN - 1 : end] = True
        ids[row, end + 1 :] = PAD_TOKEN_ID
    return ids, mask


def _experiences() -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """``learn`` input: one row per completion, two completions per prompt."""
    ids, mask = _batch()
    return list(ids.split(1)), list(mask.split(1)), torch.tensor(REWARDS)


def _turn_ids() -> torch.Tensor:
    """Two turns per row: the first half of its action tokens, then the rest."""
    _, mask = _batch()
    turn_ids = torch.full(mask.shape, -1, dtype=torch.long)
    for row, end in enumerate(ACTION_ENDS):
        start = PROMPT_LEN - 1
        split = start + (end - start + 1) // 2
        turn_ids[row, start:split] = 0
        turn_ids[row, split : end + 1] = 1
    return turn_ids


def _drift_policy(agent: GRPO) -> None:
    """Seeded LoRA ``B`` weights, so the policy differs from the base model."""
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for name, param in agent.actor.named_parameters():
            if param.requires_grad and "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.5)


def _make_agent(**overrides: Any) -> GRPO:
    """Tiny GRPO with four 2-row micro-batches in one optimizer step and a drifted policy."""
    agent = _make_grpo(
        **{
            "batch_size": NUM_ROWS // 2,
            "mini_batch_size": NUM_ROWS,
            "old_logprobs_source": "rollout",
            "vllm_importance_sampling_correction": False,
            **overrides,
        }
    )
    _drift_policy(agent)
    return agent


def _sampling_logps(agent: GRPO) -> list[torch.Tensor]:
    """Per-row rollout log-probs: the policy's own plus seeded noise."""
    ids, mask = _batch()
    _, policy, _ = agent._fused_forward_no_grad(ids, NUM_ROWS, include_reference=False)
    generator = torch.Generator().manual_seed(7)
    noise = torch.randn(policy.shape, generator=generator) * SAMPLING_NOISE_STD
    sampling = policy + noise + torch.tensor(SAMPLING_ROW_DRIFT).unsqueeze(-1)
    return [sampling[row][mask[row]] for row in range(NUM_ROWS)]


def _learn(
    monkeypatch: pytest.MonkeyPatch, use_liger_loss: bool, **overrides: Any
) -> tuple[GRPO, dict[str, float], list[dict[str, torch.Tensor]]]:
    agent = _make_agent(use_liger_loss=use_liger_loss, **overrides)
    sampling_logps = _sampling_logps(agent)
    steps = _record_step_gradients(agent, monkeypatch)
    metrics = agent.learn(_experiences(), sampling_logps=sampling_logps)
    return agent, metrics, steps


def _learn_both_paths(
    monkeypatch: pytest.MonkeyPatch,
    make_agent: Callable[[bool], GRPO],
    learn: Callable[[GRPO], dict[str, float]],
) -> list[tuple[GRPO, dict[str, float], list[dict[str, torch.Tensor]]]]:
    """Standard then fused: each agent, its learn metrics and its optimizer-step gradients."""
    runs = []
    for use_liger_loss in (False, True):
        agent = make_agent(use_liger_loss)
        steps = _record_step_gradients(agent, monkeypatch)
        runs.append((agent, learn(agent), steps))
    return runs


def _assert_same_update(
    fused: dict[str, float],
    fused_steps: list[dict[str, torch.Tensor]],
    standard: dict[str, float],
    standard_steps: list[dict[str, torch.Tensor]],
) -> None:
    """Same loss, clip fraction and per-step gradients on both paths."""
    assert fused["loss"] == pytest.approx(standard["loss"], rel=LOSS_RTOL, abs=ATOL)
    assert fused["clipfrac"] == pytest.approx(standard["clipfrac"], abs=ATOL)
    assert len(fused_steps) == len(standard_steps) > 0
    for fused_step, standard_step in zip(fused_steps, standard_steps, strict=True):
        assert fused_step.keys() == standard_step.keys()
        assert any(grad.abs().sum() > 0 for grad in standard_step.values())
        for name, grad in standard_step.items():
            assert torch.allclose(fused_step[name], grad, rtol=GRAD_RTOL, atol=ATOL), (
                name
            )


OBJECTIVES = {
    "grpo": {"loss_type": "grpo"},
    "cispo": {"loss_type": "cispo"},
    "gspo": {"loss_type": "gspo"},
}
KL_TERMS = {
    "beta0": {"beta": 0.0},
    "beta_kl_clamp": {
        "beta": 0.05,
        "kl_clamp": KL_CLAMP,
        "use_bias_correction_kl": True,
    },
}
OFF_POLICY_MASKS = {
    "unmasked": {},
    "masked": {
        "off_policy_token_mask_bounds": ICEPOP_BAND,
        "off_policy_sequence_mask_threshold": SEQUENCE_MASK_THRESHOLD,
    },
}


@pytest.mark.skipif(not HAS_LIGER_KERNEL, reason="liger-kernel is Linux-only.")
class TestGRPOLearnLigerParity:
    @pytest.mark.parametrize("masks", list(OFF_POLICY_MASKS))
    @pytest.mark.parametrize("kl_term", list(KL_TERMS))
    @pytest.mark.parametrize("objective", list(OBJECTIVES))
    @pytest.mark.parametrize(
        "loss_norm", ["micro_batch", "accumulation_window", "episode"]
    )
    def test_fused_learn_matches_the_standard_learn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        loss_norm: str,
        objective: str,
        kl_term: str,
        masks: str,
    ) -> None:
        # Arrange
        overrides = {
            "loss_norm": loss_norm,
            **OBJECTIVES[objective],
            **KL_TERMS[kl_term],
            **OFF_POLICY_MASKS[masks],
        }

        # Act
        standard_agent, standard, standard_steps = _learn(
            monkeypatch, use_liger_loss=False, **overrides
        )
        fused_agent, fused, fused_steps = _learn(
            monkeypatch, use_liger_loss=True, **overrides
        )

        # Assert: the trajectory level runs masked updates on the standard path.
        assert standard_agent._liger_path_selected is False
        assert fused_agent._liger_path_selected is not (
            objective == "gspo" and masks == "masked"
        )
        assert 0.0 < standard["clipfrac"] < 1.0
        if masks == "masked":
            assert 0.0 < standard["off_policy_token_mask_frac"] < 1.0
            assert 0.0 < standard["off_policy_seq_mask_frac"] < 1.0
        if kl_term == "beta0":
            assert math.isnan(standard["kl"])
            assert math.isnan(fused["kl"])
        else:
            assert 0.0 < standard["kl_clamp_frac"] < 1.0
            assert fused["kl"] == pytest.approx(standard["kl"], rel=LOSS_RTOL, abs=ATOL)
        assert len(standard_steps) == 1
        _assert_same_update(fused, fused_steps, standard, standard_steps)

    @pytest.mark.parametrize("objective", ["grpo", "cispo"])
    def test_turn_advantages_match_the_standard_learn(
        self, monkeypatch: pytest.MonkeyPatch, objective: str
    ) -> None:
        # Arrange: per-turn rewards give every action token its own advantage.
        ids, mask = _batch()
        experiences = (
            list(ids.split(1)),
            list(mask.split(1)),
            torch.tensor(TURN_REWARDS),
        )

        # Act
        (_, standard, standard_steps), (fused_agent, fused, fused_steps) = (
            _learn_both_paths(
                monkeypatch,
                lambda use_liger_loss: _make_agent(
                    use_liger_loss=use_liger_loss,
                    advantage_granularity="turn",
                    **OBJECTIVES[objective],
                ),
                lambda agent: agent.learn(
                    experiences,
                    turn_ids=_turn_ids(),
                    sampling_logps=_sampling_logps(agent),
                ),
            )
        )

        # Assert
        assert fused_agent._liger_path_selected is True
        _assert_same_update(fused, fused_steps, standard, standard_steps)

    @pytest.mark.parametrize("objective", ["grpo", "cispo"])
    def test_segmented_learn_matches_the_standard_learn(
        self, monkeypatch: pytest.MonkeyPatch, objective: str
    ) -> None:
        # Arrange: four episodes split into six segment rows, two optimizer steps.
        def make_agent(use_liger_loss: bool) -> GRPO:
            agent = _make_grpo(
                use_liger_loss=use_liger_loss,
                mini_batch_size=2,
                **OBJECTIVES[objective],
            )
            _drift_policy(agent)
            return agent

        # Act
        (_, standard, standard_steps), (_, fused, fused_steps) = _learn_both_paths(
            monkeypatch,
            make_agent,
            lambda agent: agent.learn(
                segment_experiences(PAD_TOKEN_ID), episode_segments=EPISODE_SEGMENTS
            ),
        )

        # Assert
        assert len(standard_steps) > 1
        _assert_same_update(fused, fused_steps, standard, standard_steps)

    @pytest.mark.parametrize("mini_batch_size", [2, 4, NUM_ROWS])
    @pytest.mark.parametrize("objective", list(OBJECTIVES))
    def test_the_default_loss_norm_matches_the_standard_learn(
        self, monkeypatch: pytest.MonkeyPatch, objective: str, mini_batch_size: int
    ) -> None:
        # Arrange: one, two or four 2-row micro-batches per optimizer step.
        overrides = {"mini_batch_size": mini_batch_size, **OBJECTIVES[objective]}

        # Act
        standard_agent, standard, standard_steps = _learn(
            monkeypatch, use_liger_loss=False, **overrides
        )
        fused_agent, fused, fused_steps = _learn(
            monkeypatch, use_liger_loss=True, **overrides
        )

        # Assert
        assert (
            standard_agent.loss_norm == fused_agent.loss_norm == "accumulation_window"
        )
        assert fused_agent._liger_path_selected is True
        assert len(standard_steps) == NUM_ROWS // mini_batch_size
        _assert_same_update(fused, fused_steps, standard, standard_steps)

    @pytest.mark.parametrize("objective", ["grpo", "cispo"])
    def test_vllm_correction_matches_the_standard_learn(
        self, monkeypatch: pytest.MonkeyPatch, objective: str
    ) -> None:
        # Arrange: trainer old log-probs, so the rollout log-probs reweight
        # tokens; the cap clamps part of them.
        overrides = {
            "old_logprobs_source": "trainer",
            "vllm_importance_sampling_correction": True,
            "vllm_importance_sampling_cap": VLLM_IS_CAP,
            **OBJECTIVES[objective],
        }

        # Act
        _, standard, standard_steps = _learn(
            monkeypatch, use_liger_loss=False, **overrides
        )
        fused_agent, fused, fused_steps = _learn(
            monkeypatch, use_liger_loss=True, **overrides
        )

        # Assert
        assert fused_agent._liger_path_selected is True
        assert 0.0 < standard["vllm_is_frac_clamped"] < 1.0
        _assert_same_update(fused, fused_steps, standard, standard_steps)

    @pytest.mark.parametrize("objective", list(OBJECTIVES))
    def test_sequence_packing_matches_the_standard_learn(
        self, monkeypatch: pytest.MonkeyPatch, objective: str
    ) -> None:
        # Arrange
        def make_agent(use_liger_loss: bool) -> GRPO:
            agent = _make_agent(
                use_liger_loss=use_liger_loss,
                use_sequence_packing=True,
                **OBJECTIVES[objective],
            )
            agent.actor.config._attn_implementation = "flash_attention_2"
            return agent

        # Act
        (_, standard, standard_steps), (fused_agent, fused, fused_steps) = (
            _learn_both_paths(
                monkeypatch,
                make_agent,
                lambda agent: agent.learn(
                    _experiences(), sampling_logps=_sampling_logps(agent)
                ),
            )
        )

        # Assert
        assert fused_agent._packing_mode() == "varlen"
        _assert_same_update(fused, fused_steps, standard, standard_steps)


class TestGRPOFusedKernelLossLigerArguments:
    """The Liger loss type, level and divisor each configuration hands the kernel."""

    @pytest.fixture
    def recording_kernel(self, monkeypatch: pytest.MonkeyPatch) -> type:
        monkeypatch.setattr(grpo_module, "HAS_LIGER_KERNEL", True)
        monkeypatch.setattr(RecordingFusedKernel, "calls", [])
        monkeypatch.setattr(
            grpo_module, KERNEL_NAME, RecordingFusedKernel, raising=False
        )
        return RecordingFusedKernel

    @staticmethod
    def _call(agent: GRPO) -> tuple[Any, ...]:
        ids, mask = _batch()
        agent._fused_kernel_loss(
            ids, mask, torch.ones(NUM_ROWS, 1), None, torch.zeros(mask.shape)
        )
        (args,) = RecordingFusedKernel.calls
        return args

    @pytest.mark.parametrize(
        ("loss_type", "liger_loss_type"), [("grpo", "dapo"), ("cispo", "cispo")]
    )
    def test_token_level_reduces_by_the_weighted_mask(
        self, recording_kernel: type, loss_type: str, liger_loss_type: str
    ) -> None:
        agent = _make_grpo(
            loss_type=loss_type, use_liger_loss=True, loss_norm="micro_batch"
        )

        args = self._call(agent)

        # Every row's weights sum to 1 / B, and the divisor is 1.
        row_weights = args[3].reshape(NUM_ROWS, -1).sum(dim=-1)
        assert args[LOSS_TYPE_ARG] == liger_loss_type
        assert args[IMPORTANCE_SAMPLING_LEVEL_ARG] == "token"
        assert args[NUM_ITEMS_ARG] == pytest.approx(1.0)
        assert torch.allclose(row_weights, torch.full((NUM_ROWS,), 1 / NUM_ROWS))

    def test_gspo_runs_liger_sequence_level_on_the_per_sequence_mean(
        self, recording_kernel: type
    ) -> None:
        agent = _make_grpo(
            loss_type="gspo",
            use_liger_loss=True,
            vllm_importance_sampling_correction=False,
            loss_norm="micro_batch",
        )

        args = self._call(agent)

        assert args[LOSS_TYPE_ARG] == "grpo"
        assert args[IMPORTANCE_SAMPLING_LEVEL_ARG] == "sequence"
        assert set(args[3].unique().tolist()) <= {0.0, 1.0}

    def test_gspo_window_divides_by_the_window_tokens(
        self, recording_kernel: type
    ) -> None:
        # Arrange: two accumulated micro-batches per optimizer step.
        agent = _make_grpo(
            loss_type="gspo",
            use_liger_loss=True,
            vllm_importance_sampling_correction=False,
            loss_norm="accumulation_window",
        )
        _, mask = _batch()
        agent._record_global_window_action_tokens(mask, np.arange(NUM_ROWS))

        # Act
        args = self._call(agent)

        # Assert: the batch's rows hold 27 action tokens.
        assert args[LOSS_TYPE_ARG] == "dapo"
        assert args[IMPORTANCE_SAMPLING_LEVEL_ARG] == "sequence"
        assert float(args[NUM_ITEMS_ARG]) == 27.0
