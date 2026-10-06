# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GRPO old log-probs and action-position scoring.

Covers where ``learn`` takes its old log-probs from (``old_logprobs_source``),
which rows the no-grad forward still scores, and log-probs computed only at
action positions. Pure CPU on a tiny real model in fp32.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from peft import LoraConfig

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.algorithms.grpo import GRPO
from tests.test_algorithms.test_llms.llm_helpers import create_module

PAD_TOKEN_ID = 63
VOCAB = 64
SEQ_LEN = 8
PROMPT_LEN = 3
NUM_ROWS = 4
GRAD_ATOL = 1e-6


def _make_grpo(**overrides: Any) -> GRPO:
    """Tiny fp32 GRPO on CPU: 2 prompts x 2 completions, one micro-batch of 2 rows."""
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
        "gradient_checkpointing": False,
        "calc_position_embeddings": False,
        "device": "cpu",
        "use_liger_loss": False,
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
    return GRPO(**kwargs)


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


class TestGRPOOldLogprobsSourceConfig:
    def test_trainer_is_the_default(self) -> None:
        assert _make_grpo().old_logprobs_source == "trainer"

    def test_unknown_source_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid old_logprobs_source 'vllm'"):
            _make_grpo(old_logprobs_source="vllm")


class TestGRPOLearnTrainerOldLogProbs:
    """``old_logprobs_source="trainer"``: old log-probs are the learn-start policy's."""

    def test_a_single_step_batch_runs_no_no_grad_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: 4 rows, 2 micro-batches, one optimizer step.
        agent = _make_grpo()
        calls = _record_no_grad_forwards(agent, monkeypatch)

        # Act
        agent.learn(_experiences())

        # Assert
        assert calls == []

    @pytest.mark.parametrize(
        ("mini_batch_size", "update_epochs"),
        [(4, 1), (2, 1), (4, 2), (2, 2)],
    )
    def test_gradients_match_an_explicit_old_forward(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mini_batch_size: int,
        update_epochs: int,
    ) -> None:
        # Arrange: the explicit agent scores every row up front.
        self_scored = _make_grpo(
            mini_batch_size=mini_batch_size, update_epochs=update_epochs
        )
        explicit = _make_grpo(
            mini_batch_size=mini_batch_size, update_epochs=update_epochs
        )
        monkeypatch.setattr(explicit, "_self_scored_micro_batches", lambda *_: 0)
        explicit_calls = _record_no_grad_forwards(explicit, monkeypatch)
        self_scored_steps = _record_step_gradients(self_scored, monkeypatch)
        explicit_steps = _record_step_gradients(explicit, monkeypatch)

        # Act
        self_scored_metrics = self_scored.learn(_experiences())
        explicit_metrics = explicit.learn(_experiences())

        # Assert
        assert explicit_calls == [(NUM_ROWS, False, True)]
        _assert_same_gradients(self_scored_steps, explicit_steps)
        assert self_scored_metrics["kl_old"] == pytest.approx(
            explicit_metrics["kl_old"], abs=1e-6
        )

    def test_rows_after_the_first_step_get_the_no_grad_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: 2 micro-batches, one optimizer step each.
        agent = _make_grpo(mini_batch_size=2)
        calls = _record_no_grad_forwards(agent, monkeypatch)

        # Act
        agent.learn(_experiences())

        # Assert
        assert calls == [(2, False, True)]

    def test_a_kl_penalty_runs_a_reference_only_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = _make_grpo(beta=0.04)
        calls = _record_no_grad_forwards(agent, monkeypatch)

        # Act
        agent.learn(_experiences())

        # Assert
        assert calls == [(NUM_ROWS, True, False)]

    def test_a_row_without_sampling_logprobs_gets_the_no_grad_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: its unit-ratio fallback needs old log-probs before the update.
        agent = _make_grpo()
        sampling_logps: list[torch.Tensor | None] = list(_sampling_logps(agent))
        sampling_logps[0] = None
        forwarded_ids: list[torch.Tensor] = []
        calls = _record_no_grad_forwards(agent, monkeypatch, forwarded_ids)
        ids, _ = _batch()

        # Act
        with pytest.warns(UserWarning, match="token-count mismatch"):
            agent.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert
        assert len(calls) == 1
        assert any(torch.equal(row, ids[0]) for row in forwarded_ids[0])

    def test_rows_without_sampling_logprobs_keep_the_full_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = _make_grpo()
        calls = _record_no_grad_forwards(agent, monkeypatch)

        # Act
        with pytest.warns(UserWarning, match="token-count mismatch"):
            agent.learn(_experiences(), sampling_logps=[None] * NUM_ROWS)

        # Assert
        assert calls == [(NUM_ROWS, False, True)]

    def test_sampling_mismatch_metrics_compare_learn_start_log_probs(self) -> None:
        # Arrange: sampling log-probs sit 0.5 below the learn-start policy.
        agent = _make_grpo()
        sampling_logps = _sampling_logps(agent, offset=-0.5)

        # Act
        metrics = agent.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert
        assert metrics["vllm_is_delta_mean"] == pytest.approx(0.5, abs=1e-5)
        assert metrics["vllm_is_ratio_mean"] == pytest.approx(
            torch.exp(torch.tensor(0.5)).item(), abs=1e-5
        )
        assert metrics["vllm_mismatch_kl"] == pytest.approx(
            math.exp(0.5) - 1.5, abs=1e-5
        )


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
        "agilerl.algorithms.grpo.allreduce_minmax_int",
        lambda value: (min(value, most_uncovered), max(value, most_uncovered)),
    )


class TestGRPOLearnRolloutOldLogProbs:
    """``old_logprobs_source="rollout"``: old log-probs are the sampling log-probs."""

    def test_no_policy_forward_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: 2 optimizer steps, so the trainer source would score 2 rows.
        agent = _make_grpo(old_logprobs_source="rollout", mini_batch_size=2)
        sampling_logps = _sampling_logps(agent)
        calls = _record_no_grad_forwards(agent, monkeypatch)

        # Act
        metrics = agent.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert
        assert calls == []
        assert metrics["old_logprobs_trainer_rows"] == 0.0
        assert "vllm_is_delta_mean" in metrics

    def test_only_uncovered_rows_get_the_policy_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = _make_grpo(old_logprobs_source="rollout")
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
        agent = _make_grpo(old_logprobs_source="rollout")
        calls = _record_no_grad_forwards(agent, monkeypatch)

        # Act
        metrics = agent.learn(_experiences())

        # Assert
        assert calls == [(NUM_ROWS, False, True)]
        assert metrics["old_logprobs_trainer_rows"] == float(NUM_ROWS)
        assert math.isfinite(metrics["loss"])

    def test_uncovered_rows_train_like_the_trainer_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: covered rows carry the learn-start policy's own log-probs, so
        # both sources see the same old log-probs only if uncovered rows get
        # the learn-start policy's too.
        rollout = _make_grpo(old_logprobs_source="rollout", mini_batch_size=2)
        trainer = _make_grpo(mini_batch_size=2)
        sampling_logps = _partly_covered_sampling_logps(trainer)
        rollout_steps = _record_step_gradients(rollout, monkeypatch)
        trainer_steps = _record_step_gradients(trainer, monkeypatch)

        # Act
        rollout_metrics = rollout.learn(_experiences(), sampling_logps=sampling_logps)
        with pytest.warns(UserWarning, match="token-count mismatch"):
            trainer_metrics = trainer.learn(
                _experiences(), sampling_logps=sampling_logps
            )

        # Assert
        _assert_same_gradients(rollout_steps, trainer_steps)
        assert math.isfinite(rollout_metrics["kl_old"])
        assert rollout_metrics["kl_old"] == pytest.approx(
            trainer_metrics["kl_old"], abs=1e-6
        )

    def test_sampling_mismatch_metrics_skip_uncovered_rows(self) -> None:
        # Arrange: covered rows sit 0.5 below the policy; one optimizer step,
        # so the update log-probs are the learn-start policy's.
        agent = _make_grpo(old_logprobs_source="rollout")
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
        agent = _make_grpo(old_logprobs_source="rollout")
        sampling_logps = _sampling_logps(agent, offset=-0.5)
        _uncovered_counts_on_other_ranks(monkeypatch, most_uncovered=2)
        calls = _record_no_grad_forwards(agent, monkeypatch)

        # Act
        metrics = agent.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert: the padding forward leaves the sampling log-probs in place.
        assert calls == [(2, False, True)]
        assert metrics["old_logprobs_trainer_rows"] == 0.0
        assert metrics["vllm_is_delta_mean"] == pytest.approx(0.5, abs=1e-5)

    def test_gradients_match_the_trainer_source_on_exact_sampling_logprobs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: two steps, so both sources reuse the old log-probs.
        rollout = _make_grpo(old_logprobs_source="rollout", mini_batch_size=2)
        trainer = _make_grpo(mini_batch_size=2)
        sampling_logps = _sampling_logps(trainer)
        rollout_steps = _record_step_gradients(rollout, monkeypatch)
        trainer_steps = _record_step_gradients(trainer, monkeypatch)

        # Act
        rollout.learn(_experiences(), sampling_logps=sampling_logps)
        trainer.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert
        _assert_same_gradients(rollout_steps, trainer_steps)

    def test_a_large_sampling_gap_logs_the_divergence_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange: sampling log-probs sit 0.5 below the policy, above the 0.1 gap limit.
        agent = _make_grpo(old_logprobs_source="rollout")
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
        agent = _make_grpo(old_logprobs_source="rollout")
        sampling_logps = _sampling_logps(agent)

        # Act
        with caplog.at_level("WARNING"):
            agent.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert
        assert "Rollout engine and trainer log-probs diverge" not in caplog.text


class TestGRPOLearnMetricKeys:
    """Every rank reports the same learn keys, whatever its sampling coverage."""

    @pytest.mark.filterwarnings("ignore:.*token-count mismatch:UserWarning")
    @pytest.mark.parametrize("source", ["trainer", "rollout"])
    def test_an_uncovered_batch_reports_the_covered_batchs_keys(
        self, source: str
    ) -> None:
        # Arrange
        covered_agent = _make_grpo(old_logprobs_source=source)
        uncovered_agent = _make_grpo(old_logprobs_source=source)
        sampling_logps = _sampling_logps(covered_agent)

        # Act
        covered = covered_agent.learn(_experiences(), sampling_logps=sampling_logps)
        uncovered = uncovered_agent.learn(
            _experiences(), sampling_logps=[None] * NUM_ROWS
        )

        # Assert
        assert list(uncovered) == list(covered)
        assert uncovered["old_logprobs_trainer_rows"] == float(NUM_ROWS)


class TestGRPORolloutActorRows:
    """Rollout-source rows that need the learn-start policy forward."""

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

    @pytest.mark.parametrize("packing_mode", [None, "varlen"])
    def test_training_forward_matches_full_scoring(
        self, monkeypatch: pytest.MonkeyPatch, packing_mode: str | None
    ) -> None:
        # Arrange
        agent = _make_grpo()
        monkeypatch.setattr(agent, "_packing_mode", lambda: packing_mode)
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
        _trained_checkpoint(tmp_path, old_logprobs_source="trainer")
        agent = _make_grpo(old_logprobs_source="rollout")

        # Act
        agent.load_checkpoint(str(tmp_path))

        # Assert
        assert agent.old_logprobs_source == "trainer"

    def test_resume_keeps_current_settings_and_restores_training_state(
        self, tmp_path
    ) -> None:
        # Arrange
        saved = _trained_checkpoint(tmp_path, old_logprobs_source="trainer")
        agent = _make_grpo(old_logprobs_source="rollout", lr=5e-3)

        # Act
        agent.load_checkpoint(str(tmp_path), restore_config=False)

        # Assert
        assert agent.old_logprobs_source == "rollout"
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
