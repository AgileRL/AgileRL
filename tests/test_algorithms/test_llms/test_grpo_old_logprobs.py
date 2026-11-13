# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GRPO old log-probs and action-position scoring.

Covers the rollout engine's sampling log-probs as old log-probs, which rows
the no-grad forward still scores, and log-probs computed only at action
positions. Pure CPU on a tiny real model in fp32.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from peft import LoraConfig

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.algorithms.core.sequence_layout import PaddedSequence
from agilerl.algorithms.grpo import GRPO
from tests.rank_launch import run_ranks
from tests.test_algorithms.test_llms.llm_helpers import create_module, use_packed

PAD_TOKEN_ID = 63
VOCAB = 64
SEQ_LEN = 8
PROMPT_LEN = 3
NUM_ROWS = 4
GRAD_ATOL = 1e-6


def _capture_sampling_logps(agent: GRPO, monkeypatch: pytest.MonkeyPatch) -> bool:
    """Start colocated vLLM inside ``get_action`` and return the capture flag."""
    captured: dict[str, bool] = {}

    def start_colocated() -> None:
        agent.colocated = True

    def fake_generate(
        prompts: Any,
        group_size: int,
        temperature: float | None,
        capture_sampling_logps: bool = False,
        capture_sampling_supports: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], None, None]:
        del temperature, capture_sampling_supports
        captured["capture"] = capture_sampling_logps
        n = len(prompts) * group_size
        token_ids = [torch.ones(1, 4, dtype=torch.long) for _ in range(n)]
        masks = [torch.ones(1, 3, dtype=torch.bool) for _ in range(n)]
        return token_ids, masks, None, None

    agent.records_sampling_supports = False
    monkeypatch.setattr(agent, "_ensure_vllm_for_generation", start_colocated)
    monkeypatch.setattr(agent, "_prepare_vllm_for_generation", lambda: None)
    monkeypatch.setattr(agent, "_generate_with_vllm_colocate", fake_generate)
    agent.get_action(
        [{"input_ids": torch.ones(1, 2, dtype=torch.long)}],
        training=True,
        repeat_prompts=False,
    )
    return captured["capture"]


def _make_grpo(algo_class: type[GRPO] = GRPO, **overrides: Any) -> GRPO:
    """Tiny fp32 GRPO-family agent on CPU: 2 prompts x 2 completions, one micro-batch of 2 rows."""
    torch.manual_seed(0)
    kwargs: dict[str, Any] = {
        "actor_network": create_module(
            input_size=6, max_tokens=4, vocab_size=VOCAB, device="cpu"
        ),
        "pad_token_id": PAD_TOKEN_ID,
        "pad_token": "<pad>",
        "batch_size": 2,
        "group_size": 2,
        "beta": 0.0,
        "lr": 1e-2,
        "max_grad_norm": None,
        "max_output_tokens": 4,
        "max_model_len": 12,
        "micro_batch_size_per_gpu": 2,
        "mini_batch_size": 4,
        "wrap": False,
        "calc_position_embeddings": False,
        "device": "cpu",
        "advantage_granularity": "trajectory",
        "lora_config": LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
        ),
        **overrides,
    }
    return algo_class(**kwargs)


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids and action masks; odd rows stop one token early and end in a pad."""
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(0, PAD_TOKEN_ID, (NUM_ROWS, SEQ_LEN), generator=generator)
    mask = torch.zeros(NUM_ROWS, SEQ_LEN - 1, dtype=torch.bool)
    for row in range(NUM_ROWS):
        end = SEQ_LEN - 1 - row % 2
        mask[row, PROMPT_LEN - 1 : end] = True
        ids[row, end + 1 :] = PAD_TOKEN_ID
    return ids, mask


def _experiences() -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """``learn`` input: one row per completion and rewards that split each group."""
    ids, mask = _batch()
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    return list(ids.split(1)), list(mask.split(1)), rewards


def _sampling_logps(agent: GRPO, offset: float = 0.0) -> list[torch.Tensor]:
    """Flat per-row log-probs of the learn-start policy, shifted by ``offset``."""
    ids, mask = _batch()
    _, actor_log_probs, _ = agent._fused_forward_no_grad(
        ids, NUM_ROWS, include_reference=False
    )
    return [actor_log_probs[row][mask[row]] + offset for row in range(NUM_ROWS)]


def _record_no_grad_forwards(
    agent: GRPO,
    monkeypatch: pytest.MonkeyPatch,
    forwarded_ids: list[torch.Tensor] | None = None,
) -> list[tuple[int, bool, bool]]:
    """Record ``(rows, reference, actor)`` for each learn-start no-grad forward.

    :param forwarded_ids: Optional list that collects each forward's token ids.
    """
    calls: list[tuple[int, bool, bool]] = []
    real: Callable[..., Any] = agent._fused_forward_no_grad

    def recording(ids: torch.Tensor, batch_size: int, **kwargs: Any) -> Any:
        if forwarded_ids is not None:
            forwarded_ids.append(ids)
        calls.append(
            (
                ids.shape[0],
                kwargs.get("include_reference", True),
                kwargs.get("include_actor", True),
            )
        )
        return real(ids, batch_size, **kwargs)

    monkeypatch.setattr(agent, "_fused_forward_no_grad", recording)
    return calls


def _record_step_gradients(
    agent: GRPO, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, torch.Tensor]]:
    """Record the trainable gradients the optimizer sees at each step."""
    steps: list[dict[str, torch.Tensor]] = []
    real_step = agent.optimizer.step

    def recording_step(*args: Any, **kwargs: Any) -> Any:
        steps.append(
            {
                name: param.grad.clone()
                for name, param in agent.actor.named_parameters()
                if param.grad is not None
            }
        )
        return real_step(*args, **kwargs)

    monkeypatch.setattr(agent.optimizer, "step", recording_step)
    return steps


def _assert_same_gradients(
    actual: list[dict[str, torch.Tensor]], expected: list[dict[str, torch.Tensor]]
) -> None:
    assert len(actual) == len(expected) > 0
    for actual_step, expected_step in zip(actual, expected, strict=True):
        assert actual_step.keys() == expected_step.keys()
        assert any(grad.abs().sum() > 0 for grad in expected_step.values())
        for name, grad in expected_step.items():
            assert torch.allclose(actual_step[name], grad, atol=GRAD_ATOL), name


class TestGRPOShouldCaptureSamplingLogps:
    def test_training_rollouts_capture_once_colocated_vllm_starts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_grpo()

        captured = _capture_sampling_logps(agent, monkeypatch)

        assert agent.colocated
        assert captured is True


class TestGRPOGetActionSamplingSupports:
    @pytest.fixture
    def agent(self, monkeypatch: pytest.MonkeyPatch) -> GRPO:
        agent = _make_grpo()
        agent.colocated = True
        agent.records_sampling_supports = True

        def fake_generate(
            prompts: Any,
            group_size: int,
            temperature: float | None,
            capture_sampling_logps: bool = False,
            capture_sampling_supports: bool = False,
        ) -> tuple[
            list[torch.Tensor], list[torch.Tensor], None, list[torch.Tensor] | None
        ]:
            n = len(prompts) * group_size
            token_ids = [torch.ones(1, 4, dtype=torch.long) for _ in range(n)]
            masks = [torch.ones(1, 3, dtype=torch.bool) for _ in range(n)]
            supports = [
                torch.tensor([[1, 2], [1, -1]], dtype=torch.int32) for _ in range(n)
            ]
            return (
                token_ids,
                masks,
                None,
                supports if capture_sampling_supports else None,
            )

        monkeypatch.setattr(agent, "_ensure_vllm_for_generation", lambda: None)
        monkeypatch.setattr(agent, "_prepare_vllm_for_generation", lambda: None)
        monkeypatch.setattr(agent, "_generate_with_vllm_colocate", fake_generate)
        return agent

    def test_training_rollouts_return_the_engine_supports(self, agent: GRPO) -> None:
        result = agent.get_action(
            [{"input_ids": torch.ones(1, 2, dtype=torch.long)}],
            training=True,
            repeat_prompts=False,
        )

        assert result.sampling_supports is not None
        assert result.sampling_supports[0].tolist() == [[1, 2], [1, -1]]

    def test_evaluation_rollouts_return_no_supports(self, agent: GRPO) -> None:
        result = agent.get_action(
            [{"input_ids": torch.ones(1, 2, dtype=torch.long)}], training=False
        )

        assert result.sampling_supports is None


def _partly_covered_sampling_logps(
    agent: GRPO, offset: float = 0.0
) -> list[torch.Tensor | None]:
    """Sampling log-probs with row 1 missing and row 2 one token short."""
    sampling_logps: list[torch.Tensor | None] = list(_sampling_logps(agent, offset))
    sampling_logps[1] = None
    row_2 = sampling_logps[2]
    assert row_2 is not None
    sampling_logps[2] = row_2[:-1]
    return sampling_logps


def _uncovered_counts_on_other_ranks(
    monkeypatch: pytest.MonkeyPatch, most_uncovered: int
) -> None:
    """Make the cross-rank all-reduce report ``most_uncovered`` on another rank."""
    monkeypatch.setattr(
        "agilerl.algorithms.core.base.allreduce_minmax_int",
        lambda value: (min(value, most_uncovered), max(value, most_uncovered)),
    )


class TestGRPOLearnOldLogProbs:
    """Old log-probs are the sampling log-probs where a row has them."""

    def test_no_policy_forward_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: 2 optimizer steps.
        agent = _make_grpo(mini_batch_size=2)
        sampling_logps = _sampling_logps(agent)
        calls = _record_no_grad_forwards(agent, monkeypatch)

        # Act
        metrics = agent.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert
        assert calls == []
        assert metrics["old_logprobs_trainer_rows"] == 0.0
        assert "vllm_is_delta_mean" in metrics

    def test_a_kl_penalty_runs_a_reference_only_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = _make_grpo(beta=0.04)
        sampling_logps = _sampling_logps(agent)
        calls = _record_no_grad_forwards(agent, monkeypatch)

        # Act
        agent.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert
        assert calls == [(NUM_ROWS, True, False)]

    def test_only_uncovered_rows_get_the_policy_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = _make_grpo()
        sampling_logps = _partly_covered_sampling_logps(agent)
        forwarded_ids: list[torch.Tensor] = []
        calls = _record_no_grad_forwards(agent, monkeypatch, forwarded_ids)
        ids, _ = _batch()

        # Act
        metrics = agent.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert
        assert calls == [(2, False, True)]
        assert torch.equal(forwarded_ids[0], ids[[1, 2]])
        assert metrics["old_logprobs_trainer_rows"] == 2.0

    def test_missing_sampling_logprobs_score_every_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = _make_grpo()
        calls = _record_no_grad_forwards(agent, monkeypatch)

        # Act
        metrics = agent.learn(_experiences())

        # Assert
        assert calls == [(NUM_ROWS, False, True)]
        assert metrics["old_logprobs_trainer_rows"] == float(NUM_ROWS)
        assert math.isfinite(metrics["loss"])

    def test_uncovered_rows_train_like_covered_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: sampling log-probs are the learn-start policy's own, so
        # uncovered rows match only if they get the learn-start policy's too.
        partly_covered = _make_grpo(mini_batch_size=2)
        covered = _make_grpo(mini_batch_size=2)
        partly_covered_steps = _record_step_gradients(partly_covered, monkeypatch)
        covered_steps = _record_step_gradients(covered, monkeypatch)

        # Act
        partly_covered_metrics = partly_covered.learn(
            _experiences(),
            sampling_logps=_partly_covered_sampling_logps(partly_covered),
        )
        covered_metrics = covered.learn(
            _experiences(), sampling_logps=_sampling_logps(covered)
        )

        # Assert
        _assert_same_gradients(partly_covered_steps, covered_steps)
        assert math.isfinite(partly_covered_metrics["kl_old"])
        assert partly_covered_metrics["kl_old"] == pytest.approx(
            covered_metrics["kl_old"], abs=1e-6
        )

    def test_sampling_mismatch_metrics_skip_uncovered_rows(self) -> None:
        # Arrange: covered rows sit 0.5 below the policy; one optimizer step,
        # so the update log-probs are the learn-start policy's.
        agent = _make_grpo()
        sampling_logps = _partly_covered_sampling_logps(agent, offset=-0.5)

        # Act
        metrics = agent.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert
        assert metrics["vllm_is_delta_mean"] == pytest.approx(0.5, abs=1e-5)
        assert metrics["vllm_is_delta_max"] == pytest.approx(0.5, abs=1e-5)

    def test_a_covered_rank_matches_another_ranks_policy_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: this rank is fully covered; another rank lacks 2 rows.
        agent = _make_grpo()
        sampling_logps = _sampling_logps(agent, offset=-0.5)
        _uncovered_counts_on_other_ranks(monkeypatch, most_uncovered=2)
        calls = _record_no_grad_forwards(agent, monkeypatch)

        # Act
        metrics = agent.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert: the padding forward leaves the sampling log-probs in place.
        assert calls == [(2, False, True)]
        assert metrics["old_logprobs_trainer_rows"] == 0.0
        assert metrics["vllm_is_delta_mean"] == pytest.approx(0.5, abs=1e-5)

    def test_a_large_sampling_gap_logs_the_divergence_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange: sampling log-probs sit 0.5 below the policy, above the 0.1 gap limit.
        agent = _make_grpo()
        sampling_logps = _sampling_logps(agent, offset=-0.5)

        # Act
        with caplog.at_level("WARNING"):
            metrics = agent.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert
        assert metrics["vllm_is_delta_mean"] == pytest.approx(0.5, abs=1e-5)
        assert metrics["vllm_mismatch_kl"] == pytest.approx(
            math.exp(0.5) - 1.5, abs=1e-5
        )
        assert "Rollout engine and trainer log-probs diverge" in caplog.text

    def test_matching_sampling_logprobs_log_no_divergence_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        agent = _make_grpo()
        sampling_logps = _sampling_logps(agent)

        # Act
        with caplog.at_level("WARNING"):
            agent.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert
        assert "Rollout engine and trainer log-probs diverge" not in caplog.text


def _actor_log_softmax(agent: GRPO) -> torch.Tensor:
    """``(B, T - 1, V)`` learn-start policy log-softmax from a plain actor forward."""
    ids, _ = _batch()
    with torch.no_grad(), agent.select_adapter("actor"):
        logits = agent.actor(input_ids=ids).logits
    return (logits[:, :-1] / agent.temperature).log_softmax(-1)


def _supported_sampling(
    agent: GRPO, width: int = 3
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-row supports and the learn-start policy's log-probs over them.

    Each action token's support is the token and its likeliest rivals, padded
    with ``-1``; each row's last action token is forced (a support of one).
    """
    ids, mask = _batch()
    log_probs = _actor_log_softmax(agent)
    supports, logps = [], []
    for row in range(NUM_ROWS):
        positions = mask[row].nonzero().squeeze(1).tolist()
        row_supports = torch.full((len(positions), width), -1, dtype=torch.int32)
        row_logps = torch.zeros(len(positions))
        for index, position in enumerate(positions):
            target = int(ids[row, position + 1])
            allowed = [target]
            if index < len(positions) - 1:
                ranked = log_probs[row, position].topk(width).indices.tolist()
                allowed += [token for token in ranked if token != target][: width - 1]
            row_supports[index, : len(allowed)] = torch.tensor(allowed)
            position_log_probs = log_probs[row, position]
            row_logps[index] = position_log_probs[target] - position_log_probs[
                allowed
            ].logsumexp(0)
        supports.append(row_supports)
        logps.append(row_logps)
    return supports, logps


class TestGRPOLearnSamplingSupports:
    """Log-probs normalized over the support vLLM sampled each token from."""

    def test_old_logprobs_match_the_policy_over_the_support(self) -> None:
        # Arrange: one optimizer step, so the update policy is the learn-start one.
        agent = _make_grpo()
        supports, sampling_logps = _supported_sampling(agent)

        # Act
        metrics = agent.learn(
            _experiences(), sampling_logps=sampling_logps, sampling_supports=supports
        )

        # Assert
        assert metrics["kl_old"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["vllm_is_delta_mean"] == pytest.approx(0.0, abs=1e-5)
        assert metrics["vllm_is_delta_max"] == pytest.approx(0.0, abs=1e-5)

    def test_penalised_rollout_logprobs_give_the_exact_mismatch(
        self,
    ) -> None:
        # Arrange: vLLM applies the penalty before top-k, then reports the
        # token's log-prob under that penalised distribution.
        agent = _make_grpo()
        ids, mask = _batch()
        log_probs = _actor_log_softmax(agent)
        penalty, width = 0.7, 3
        supports, sampling_logps, expected_deltas = [], [], []
        for row in range(NUM_ROWS):
            positions = mask[row].nonzero().squeeze(1).tolist()
            row_supports = torch.full((len(positions), width), -1, dtype=torch.int32)
            row_logps = torch.zeros(len(positions))
            for index, position in enumerate(positions):
                target = int(ids[row, position + 1])
                seen = torch.zeros(VOCAB, dtype=torch.bool)
                seen[ids[row, : position + 1]] = True
                penalised = log_probs[row, position] - penalty * seen.float()
                ranked = penalised.topk(width).indices.tolist()
                rivals = [token for token in ranked if token != target]
                allowed = [target, *rivals[: width - 1]]
                row_supports[index] = torch.tensor(allowed)
                behaviour = penalised[target] - penalised[allowed].logsumexp(0)
                policy = log_probs[row, position, target] - log_probs[
                    row, position, allowed
                ].logsumexp(0)
                row_logps[index] = behaviour
                expected_deltas.append(float(policy - behaviour))
            supports.append(row_supports)
            sampling_logps.append(row_logps)
        deltas = torch.tensor(expected_deltas)

        # Act
        metrics = agent.learn(
            _experiences(), sampling_logps=sampling_logps, sampling_supports=supports
        )

        # Assert
        assert deltas.abs().max() > 0.05
        assert metrics["vllm_is_delta_mean"] == pytest.approx(
            float(deltas.abs().mean()), abs=1e-5
        )
        assert metrics["vllm_is_ratio_mean"] == pytest.approx(
            float(deltas.exp().clamp(max=agent.vllm_importance_sampling_cap).mean()),
            abs=1e-5,
        )

    def test_full_vocabulary_scoring_diverges_from_support_logprobs(self) -> None:
        agent = _make_grpo()
        _supports, sampling_logps = _supported_sampling(agent)

        metrics = agent.learn(_experiences(), sampling_logps=sampling_logps)

        assert metrics["vllm_is_delta_mean"] > 1.0

    @pytest.mark.parametrize("packed", [False, True])
    def test_scored_logprobs_are_normalized_over_the_support(
        self, packed: bool
    ) -> None:
        # Arrange
        agent = _make_grpo()
        if packed:
            use_packed(agent, False)
        ids, mask = _batch()
        supports, expected = _supported_sampling(agent)
        support_ids = GRPO._align_sampling_supports(supports, ids, mask)

        # Act
        scored = agent._get_logprobs(
            ids, batch_size=2, score_mask=mask, support_ids=support_ids
        )

        # Assert
        assert torch.allclose(scored[mask], torch.cat(expected), atol=1e-5)

    def test_a_token_outside_its_support_raises(self) -> None:
        agent = _make_grpo()
        supports, sampling_logps = _supported_sampling(agent)
        supports[2][0] = torch.tensor([PAD_TOKEN_ID, -1, -1], dtype=torch.int32)

        with pytest.raises(ValueError, match="outside the sampling support"):
            agent.learn(
                _experiences(),
                sampling_logps=sampling_logps,
                sampling_supports=supports,
            )


class TestLLMAlgorithmAlignSamplingSupports:
    """Per-row sampling supports scattered onto the action frame."""

    def test_single_token_supports_get_a_two_wide_frame(self) -> None:
        # Arrange
        agent = _make_grpo()
        ids, mask = _batch()
        supports = [
            ids[row, 1:][mask[row]].to(torch.int32).unsqueeze(-1)
            for row in range(NUM_ROWS)
        ]

        # Act
        frame = GRPO._align_sampling_supports(supports, ids, mask)
        scored = agent._get_logprobs(
            ids, batch_size=2, score_mask=mask, support_ids=frame
        )

        # Assert
        assert frame.shape == (NUM_ROWS, SEQ_LEN - 1, 2)
        assert torch.equal(frame[..., 0][mask], ids[:, 1:][mask].to(torch.int32))
        assert (frame[..., 0][~mask] == -1).all()
        assert (frame[..., 1] == -1).all()
        assert torch.allclose(scored[mask], torch.zeros(int(mask.sum())))

    def test_a_row_without_supports_keeps_a_padded_frame(self) -> None:
        # Arrange
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mask = torch.tensor([[True, True], [True, True]])
        supports = [torch.tensor([[2, 7], [3, -1]], dtype=torch.int32), None]

        # Act
        frame = GRPO._align_sampling_supports(supports, ids, mask)

        # Assert
        assert frame.tolist() == [[[2, 7], [3, -1]], [[-1, -1], [-1, -1]]]


class TestGRPOLearnMetricKeys:
    """Every rank reports the same learn keys, whatever its sampling coverage."""

    def test_an_uncovered_batch_reports_the_covered_batchs_keys(self) -> None:
        # Arrange
        covered_agent = _make_grpo()
        uncovered_agent = _make_grpo()
        sampling_logps = _sampling_logps(covered_agent)

        # Act
        covered = covered_agent.learn(_experiences(), sampling_logps=sampling_logps)
        uncovered = uncovered_agent.learn(
            _experiences(), sampling_logps=[None] * NUM_ROWS
        )

        # Assert
        assert list(uncovered) == list(covered)
        assert uncovered["old_logprobs_trainer_rows"] == float(NUM_ROWS)

    @pytest.mark.multi_rank
    def test_a_rank_without_sampling_logprobs_reports_every_ranks_keys(self) -> None:
        run_ranks(_one_rank_captured_worker)


class TestGRPORolloutActorRows:
    """Rows that need the learn-start policy forward."""

    def test_uncovered_rows_alone_when_no_rank_has_more(self) -> None:
        # Arrange
        rollout_rows = np.array([True, False, True, False])

        # Act
        rows = GRPO._rollout_actor_rows(rollout_rows)

        # Assert
        assert rows.tolist() == [1, 3]

    def test_every_rank_fully_covered_needs_no_rows(self) -> None:
        rows = GRPO._rollout_actor_rows(np.ones(NUM_ROWS, dtype=bool))

        assert rows.tolist() == []

    def test_covered_rows_pad_up_to_the_largest_rank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: another rank lacks sampling log-probs on 3 rows.
        _uncovered_counts_on_other_ranks(monkeypatch, most_uncovered=3)
        rollout_rows = np.array([True, True, False, True])

        # Act
        rows = GRPO._rollout_actor_rows(rollout_rows)

        # Assert
        assert rows.tolist() == [0, 1, 2]


class TestScoredLogProbs:
    """Log-probs scored only at ``score_mask`` match full scoring there."""

    @pytest.mark.parametrize("packed", [False, True])
    def test_training_forward_matches_full_scoring(self, packed: bool) -> None:
        # Arrange
        agent = _make_grpo()
        if not packed:
            agent.sequence_layout = PaddedSequence(
                calc_position_embeddings=agent.calc_position_embeddings,
            )
        else:
            use_packed(agent, False)
        ids, mask = _batch()

        # Act
        full = agent._get_logprobs(ids, batch_size=2)
        scored = agent._get_logprobs(ids, batch_size=2, score_mask=mask)

        # Assert
        assert scored.shape == full.shape
        assert scored.requires_grad
        assert torch.allclose(scored[mask], full[mask], atol=1e-5)
        assert torch.equal(scored[~mask], torch.zeros_like(scored[~mask]))

    def test_scored_positions_carry_the_full_gradient(self) -> None:
        # Arrange
        full_agent = _make_grpo()
        scored_agent = _make_grpo()
        ids, mask = _batch()
        weights = torch.linspace(-1.0, 1.0, mask.numel()).reshape(mask.shape) * mask

        # Act
        (full_agent._get_logprobs(ids, batch_size=2) * weights).sum().backward()
        (
            scored_agent._get_logprobs(ids, batch_size=2, score_mask=mask) * weights
        ).sum().backward()

        # Assert
        full_grads = {
            name: param.grad
            for name, param in full_agent.actor.named_parameters()
            if param.grad is not None
        }
        scored_grads = {
            name: param.grad
            for name, param in scored_agent.actor.named_parameters()
            if param.grad is not None
        }
        _assert_same_gradients([scored_grads], [full_grads])

    def test_no_grad_forward_matches_full_scoring(self) -> None:
        # Arrange
        agent = _make_grpo()
        ids, mask = _batch()

        # Act
        full_ref, full_actor, _ = agent._fused_forward_no_grad(ids, 2)
        scored_ref, scored_actor, _ = agent._fused_forward_no_grad(
            ids, 2, score_mask=mask
        )

        # Assert
        for scored, full in ((scored_ref, full_ref), (scored_actor, full_actor)):
            assert torch.allclose(scored[mask], full[mask], atol=1e-5)
            assert torch.equal(scored[~mask], torch.zeros_like(scored[~mask]))

    def test_reference_only_forward_leaves_the_policy_unscored(self) -> None:
        # Arrange
        agent = _make_grpo()
        ids, mask = _batch()

        # Act
        full_ref, _, _ = agent._fused_forward_no_grad(ids, 2)
        ref, actor, _ = agent._fused_forward_no_grad(
            ids, 2, include_actor=False, score_mask=mask
        )

        # Assert
        assert torch.allclose(ref[mask], full_ref[mask], atol=1e-5)
        assert torch.isnan(actor).all()


def _lora_weights(agent: GRPO) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in agent.actor.named_parameters()
        if "lora_" in name
    }


def _trained_checkpoint(path: Path, **overrides: Any) -> GRPO:
    """Save a GRPO with non-default LoRA weights and training counters."""
    agent = _make_grpo(**overrides)
    with torch.no_grad():
        for name, param in agent.actor.named_parameters():
            if "lora_B" in name:
                param.fill_(0.25)
    agent.steps = 7
    agent.reference_update_tracker = 3
    agent.save_checkpoint(str(path))
    return agent


class TestGRPOLearnSequencePacking:
    """A packed gradient forward trains exactly as the padded one."""

    @pytest.mark.parametrize("loss_norm", ["accumulation_window", "micro_batch"])
    def test_packed_learn_matches_padded(
        self, monkeypatch: pytest.MonkeyPatch, loss_norm: str
    ) -> None:
        # Arrange: two micro-batches per step, odd rows padded.
        padded = _make_grpo(loss_norm=loss_norm, mini_batch_size=4)
        packed = _make_grpo(loss_norm=loss_norm, mini_batch_size=4)
        use_packed(packed, True)
        padded_steps = _record_step_gradients(padded, monkeypatch)
        packed_steps = _record_step_gradients(packed, monkeypatch)

        # Act
        padded_metrics = padded.learn(
            _experiences(), sampling_logps=_sampling_logps(padded)
        )
        packed_metrics = packed.learn(
            _experiences(), sampling_logps=_sampling_logps(packed)
        )

        # Assert
        assert padded_metrics["padding_frac_after_packing"] > 0.0
        assert packed_metrics["padding_frac_after_packing"] == 0.0
        assert packed_metrics["loss"] == pytest.approx(padded_metrics["loss"], abs=1e-6)
        _assert_same_gradients(packed_steps, padded_steps)


class TestGRPOLoadCheckpoint:
    """What ``load_checkpoint`` takes from the checkpoint vs the loading agent."""

    def test_keeps_the_loading_agents_metrics_tracker(self, tmp_path) -> None:
        # Arrange
        _trained_checkpoint(tmp_path)
        agent = _make_grpo()
        tracker = agent.metrics
        agent.metrics.register("added_metric")

        # Act
        agent.load_checkpoint(str(tmp_path))
        agent.metrics.log("added_metric", 1.0)

        # Assert
        assert agent.metrics is tracker
        assert agent.metrics.get_mean("added_metric") == 1.0
        assert agent.steps == 7

    def test_restores_checkpoint_settings_by_default(self, tmp_path) -> None:
        # Arrange
        _trained_checkpoint(tmp_path, temperature=0.5)
        agent = _make_grpo(temperature=0.9)

        # Act
        agent.load_checkpoint(str(tmp_path))

        # Assert
        assert agent.temperature == 0.5

    def test_resume_keeps_current_settings_and_restores_training_state(
        self, tmp_path
    ) -> None:
        # Arrange
        saved = _trained_checkpoint(tmp_path, temperature=0.5)
        agent = _make_grpo(temperature=0.9, lr=5e-3)

        # Act
        agent.load_checkpoint(str(tmp_path), restore_config=False)

        # Assert
        assert agent.temperature == 0.9
        assert agent.lr == 5e-3
        assert agent.steps == 7
        assert agent.reference_update_tracker == 3
        loaded = _lora_weights(agent)
        for name, weight in _lora_weights(saved).items():
            assert torch.equal(loaded[name], weight), name

    def test_resume_restores_registry_hyperparameters(self, tmp_path) -> None:
        # Arrange
        hp_config = HyperparameterConfig(lr=RLParameter(min=1e-4, max=1e-1))
        _trained_checkpoint(tmp_path, hp_config=hp_config, lr=1e-2)
        agent = _make_grpo(hp_config=hp_config, lr=5e-3)

        # Act
        agent.load_checkpoint(str(tmp_path), restore_config=False)

        # Assert
        assert agent.lr == 1e-2
        assert {group["lr"] for group in agent.optimizer.param_groups} == {1e-2}

    def test_resume_keeps_hyperparameters_when_not_restored(self, tmp_path) -> None:
        # Arrange
        hp_config = HyperparameterConfig(lr=RLParameter(min=1e-4, max=1e-1))
        _trained_checkpoint(tmp_path, hp_config=hp_config, lr=1e-2)
        agent = _make_grpo(hp_config=hp_config, lr=5e-3)

        # Act
        agent.load_checkpoint(
            str(tmp_path),
            load_optimizer=True,
            restore_config=False,
            restore_hyperparameters=False,
        )

        # Assert
        assert agent.lr == 5e-3
        assert {group["lr"] for group in agent.optimizer.param_groups} == {5e-3}
        assert agent.steps == 7


def _one_rank_captured_worker(rank: int, world_size: int) -> None:
    # Arrange: four rows per rank; only rank 0 captured sampling log-probs.
    agent = _make_grpo(batch_size=2 * world_size)
    sampling_logps = _sampling_logps(agent, offset=-0.5) if rank == 0 else None

    # Act
    metrics = agent.learn(_experiences(), sampling_logps=sampling_logps)
    every_rank_keys: list[list[str] | None] = [None] * world_size
    dist.all_gather_object(every_rank_keys, list(metrics))

    # Assert
    assert every_rank_keys[0] == every_rank_keys[1]
    assert "vllm_is_delta_mean" in every_rank_keys[0]
    if rank == 1:
        assert metrics["old_logprobs_trainer_rows"] == float(NUM_ROWS)
