# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the shared :meth:`LLMAlgorithm.test` evaluation loop.

The method only touches ``self.get_action``, ``self.metrics`` and
``self.distributed`` (the end-of-eval barrier), so it is exercised here with a
stub in place of a fully constructed algorithm — no model, GPU, or vLLM
required.
"""

import logging
import math

import numpy as np
import pytest
import torch

from agilerl.algorithms.core import ActionResult
from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.llm_envs import RolloutHarness
from agilerl.metrics import AgentMetrics
from tests.helpers.rollout_doubles import FakeEnvClient


class _StubAlgo:
    """Minimal stand-in exposing the attributes ``LLMAlgorithm.test`` uses."""

    def __init__(self):
        self.metrics = AgentMetrics()
        self.distributed = False

    @property
    def fitness(self):
        return list(self.metrics.fitness)

    def get_action(self, prompts, training=False):
        del prompts, training
        completion = torch.ones(1, 5, dtype=torch.long)
        action_mask = torch.ones(1, 4, dtype=torch.bool)
        return ActionResult([completion], [action_mask])


class _TinyTokenizer:
    """Raw-encoding tokenizer for the ``apply_chat_template=False`` paths."""

    def __call__(self, texts, **kwargs):
        del kwargs
        return {"input_ids": torch.ones(len(texts), 4, dtype=torch.long)}

    def encode(self, text, add_special_tokens=True):
        del text, add_special_tokens
        return [7, 8]

    def decode(self, ids, skip_special_tokens=True):
        del ids, skip_special_tokens
        return "gen"


def _rollout_env(client: FakeEnvClient, max_turns: int, **kwargs) -> RolloutHarness:
    return RolloutHarness(
        client,
        _TinyTokenizer(),
        max_turns=max_turns,
        apply_chat_template=False,
        **kwargs,
    )


class TestLLMAlgorithmTest:
    def test_env_truncation_ends_each_episode_at_max_turns(self):
        algo = _StubAlgo()
        client = FakeEnvClient()  # backend never ends the episode itself
        env = _rollout_env(client, max_turns=3)

        out = LLMAlgorithm.test(algo, env, loop=2)

        # Two episodes of exactly ``max_turns`` turns each, run under eval mode.
        assert client.step_calls == 2 * env.max_turns
        assert client.eval_mode_entries == 1
        assert isinstance(out, np.ndarray)
        assert out.shape == ()
        assert out.item() == pytest.approx(1.0)
        assert algo.fitness == [pytest.approx(1.0)]

    def test_terminating_env_finishes_before_max_turns(self):
        algo = _StubAlgo()
        client = FakeEnvClient(reward=0.5, terminate_after=2)
        env = _rollout_env(client, max_turns=5)

        out = LLMAlgorithm.test(algo, env, loop=1)

        assert client.step_calls == 2
        assert out.item() == pytest.approx(0.5)

    def test_done_at_reset_records_zero_fitness(self):
        """An over-budget initial prompt ends the episode before any turn."""
        algo = _StubAlgo()
        env = _rollout_env(
            FakeEnvClient(),
            max_turns=3,
            max_model_len=4,
        )

        with pytest.warns(UserWarning, match="collected no turns"):
            out = LLMAlgorithm.test(algo, env, loop=1)

        assert out.item() == 0.0
        assert algo.fitness == [0.0]

    def test_subclasses_share_the_base_implementation(self):
        from agilerl.algorithms.grpo import GRPO
        from agilerl.algorithms.ppo_llm import PPO
        from agilerl.algorithms.reinforce_llm import REINFORCE

        assert GRPO.test is LLMAlgorithm.test
        assert PPO.test is LLMAlgorithm.test
        assert REINFORCE.test is LLMAlgorithm.test


class TestCreatePromptMasks:
    def test_first_response_token_is_included(self):
        # Prompt occupies positions [0, P); the first response token sits AT
        # index P and must be trainable — a strict ``>`` drops the
        # highest-information token of every SFT/DPO example.
        masks = LLMAlgorithm._create_prompt_masks([2, 3], max_length=5)
        assert masks.tolist() == [
            [False, False, True, True, True],
            [False, False, False, True, True],
        ]


class TestLLMAlgorithmSegmentLossScales:
    def test_single_rank_steps_average_their_real_micro_batches(self) -> None:
        # Arrange: two 4-micro-batch steps with 3 and 4 real micro-batches.
        filler = np.array([False, True, False, False, False, False, False, False])

        # Act
        scales = LLMAlgorithm._segment_loss_scales(filler, 4)

        # Assert
        assert scales.tolist() == pytest.approx([4 / 3, 0.0, 4 / 3, 4 / 3, 1, 1, 1, 1])

    def test_steps_average_the_real_micro_batches_of_every_rank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: the other rank's two steps hold 1 and 2 real micro-batches.
        monkeypatch.setattr("agilerl.algorithms.core.base.get_world_size", lambda: 2)
        monkeypatch.setattr(
            "agilerl.algorithms.core.base.allreduce_sum_ints",
            lambda values: [values[0] + 1, values[1] + 2],
        )
        filler = np.array([False, True, True, False])

        # Act
        scales = LLMAlgorithm._segment_loss_scales(filler, 2)

        # Assert
        assert scales.tolist() == pytest.approx([2.0, 0.0, 0.0, 4 / 3])

    def test_a_step_of_filler_on_every_rank_has_zero_scales(self) -> None:
        scales = LLMAlgorithm._segment_loss_scales(np.array([True, True]), 2)

        assert scales.tolist() == [0.0, 0.0]


class TestLLMAlgorithmTestPromptGuard:
    def test_a_non_terminal_env_holding_no_prompt_is_rejected(self):
        """``done`` and ``current_prompt`` must agree, or get_action sees nothing.

        The loop reads the prompt only while the env says it is still running,
        so a live env holding an empty prompt is a contract break, not an
        episode that quietly generates from nothing.
        """
        algo = _StubAlgo()
        env = _rollout_env(FakeEnvClient(), max_turns=2)

        class _NeverDoneEnv(RolloutHarness):
            done = False

            def reset(self, *args, **kwargs):
                del args, kwargs
                return {}, {}

        env.__class__ = _NeverDoneEnv

        with pytest.raises(TypeError, match="always holds a prompt"):
            LLMAlgorithm.test(algo, env, loop=1)


class SamplingMismatchStub:
    """Carries the state the vLLM sampling-mismatch path reads."""

    def __init__(
        self, *, max_logprob_gap: float = 0.1, max_clip_fraction: float = 0.02
    ) -> None:
        self.vllm_importance_sampling_cap = 2.0
        self.vllm_max_logprob_gap = max_logprob_gap
        self.vllm_max_clip_fraction = max_clip_fraction

    _align_sampling_logprobs = LLMAlgorithm._align_sampling_logprobs
    _sampling_mismatch_metrics = LLMAlgorithm._sampling_mismatch_metrics
    _warn_on_sampling_mismatch = LLMAlgorithm._warn_on_sampling_mismatch
    _aligned_sampling_logprobs_and_metrics = (
        LLMAlgorithm._aligned_sampling_logprobs_and_metrics
    )


class TestLLMAlgorithmSamplingMismatchMetrics:
    """Detached engine-vs-trainer divergence statistics."""

    def test_reports_k3_mismatch_kl_over_action_tokens(self) -> None:
        # Arrange: log-diffs 0 and ln 2 give k3 terms 0 and 1 - ln 2; the
        # masked third token would add e^5 - 6 if it were counted.
        stub = SamplingMismatchStub()
        masks = torch.tensor([[True, True, False]])
        old = torch.tensor([[-1.0, -1.0, 0.0]])
        sampling = torch.tensor([[-1.0, -1.0 - math.log(2.0), -5.0]])

        # Act
        metrics = stub._sampling_mismatch_metrics(old, sampling, masks)

        # Assert
        assert metrics["vllm_mismatch_kl"] == pytest.approx(
            (1.0 - math.log(2.0)) / 2, rel=1e-6
        )

    def test_identical_log_probs_give_zero_mismatch_kl(self) -> None:
        stub = SamplingMismatchStub()
        masks = torch.ones(1, 3, dtype=torch.bool)
        log_probs = torch.tensor([[-0.5, -2.0, -0.1]])

        metrics = stub._sampling_mismatch_metrics(log_probs, log_probs.clone(), masks)

        assert metrics["vllm_mismatch_kl"] == 0.0

    def test_no_action_tokens_report_neutral_values_under_the_same_keys(
        self,
    ) -> None:
        # Arrange
        stub = SamplingMismatchStub()
        log_probs = torch.tensor([[-0.5, -2.0]])
        covered = stub._sampling_mismatch_metrics(
            log_probs, log_probs - 1.0, torch.ones(1, 2, dtype=torch.bool)
        )

        # Act
        empty = stub._sampling_mismatch_metrics(
            log_probs, log_probs - 1.0, torch.zeros(1, 2, dtype=torch.bool)
        )

        # Assert
        assert list(empty) == list(covered)
        assert empty == {
            "vllm_is_delta_mean": 0.0,
            "vllm_mismatch_kl": 0.0,
            "vllm_is_delta_max": 0.0,
            "vllm_is_ratio_mean": 1.0,
            "vllm_is_ratio_std": 0.0,
            "vllm_is_ratio_p95": 1.0,
            "vllm_is_frac_clamped": 0.0,
        }


class TestLLMAlgorithmAlignedSamplingLogprobsAndMetrics:
    """Engine-vs-trainer mismatch metrics and the warning guard on them."""

    def test_reports_ratio_std_over_action_tokens(self) -> None:
        # Arrange: log-diffs 0 and ln(1.5) give ratios 1.0 and 1.5.
        stub = SamplingMismatchStub(max_logprob_gap=1.0)
        masks = torch.ones(1, 2, dtype=torch.bool)
        old = torch.zeros(1, 2)
        sampling = [torch.tensor([0.0, -math.log(1.5)])]

        # Act
        _, metrics = stub._aligned_sampling_logprobs_and_metrics(sampling, masks, old)

        # Assert
        assert metrics["vllm_is_ratio_mean"] == pytest.approx(1.25)
        assert metrics["vllm_is_ratio_std"] == pytest.approx(0.25)

    def test_mismatch_within_limits_logs_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange: a 0.02 nat gap on every token, ratio e^0.02 under the cap.
        stub = SamplingMismatchStub()
        masks = torch.ones(1, 3, dtype=torch.bool)
        old = torch.full((1, 3), -1.0)
        sampling = [torch.full((3,), -1.02)]

        # Act
        with caplog.at_level(logging.WARNING, logger="agilerl.algorithms.core.base"):
            stub._aligned_sampling_logprobs_and_metrics(sampling, masks, old)

        # Assert
        assert caplog.records == []

    def test_gap_over_limit_logs_the_mismatch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange: a 0.5 nat gap on every token, ratio e^0.5 under the cap.
        stub = SamplingMismatchStub()
        masks = torch.ones(1, 3, dtype=torch.bool)
        old = torch.full((1, 3), -1.0)
        sampling = [torch.full((3,), -1.5)]

        # Act
        with caplog.at_level(logging.WARNING, logger="agilerl.algorithms.core.base"):
            stub._aligned_sampling_logprobs_and_metrics(sampling, masks, old)

        # Assert
        assert [record.getMessage() for record in caplog.records] == [
            (
                "Rollout engine and trainer log-probs diverge: mean |gap| 0.5000 "
                "(max 0.1000), ratio mean 1.6487 std 0.0000, clip fraction 0.0000 "
                "(max 0.0200) at cap 2.00."
            )
        ]

    def test_clip_fraction_over_limit_logs_the_mismatch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange: one of four tokens has a 1.0 nat gap, so its ratio e^1.0
        # clamps at the cap of 2.0; the 0.25 mean gap stays under its limit.
        stub = SamplingMismatchStub(max_logprob_gap=1.0)
        masks = torch.ones(1, 4, dtype=torch.bool)
        old = torch.zeros(1, 4)
        sampling = [torch.tensor([-1.0, 0.0, 0.0, 0.0])]

        # Act
        with caplog.at_level(logging.WARNING, logger="agilerl.algorithms.core.base"):
            stub._aligned_sampling_logprobs_and_metrics(sampling, masks, old)

        # Assert
        assert [record.getMessage() for record in caplog.records] == [
            (
                "Rollout engine and trainer log-probs diverge: mean |gap| 0.2500 "
                "(max 1.0000), ratio mean 1.2500 std 0.4330, clip fraction 0.2500 "
                "(max 0.0200) at cap 2.00."
            )
        ]
