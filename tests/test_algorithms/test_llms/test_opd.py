# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for on-policy distillation (:class:`agilerl.algorithms.opd.OPD`).

Pure CPU on a tiny real model in fp32.
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl.algorithms import opd as opd_module
from agilerl.algorithms.grpo import GRPO
from agilerl.algorithms.opd import OPD, _repeated_ngram_frac
from tests.test_algorithms.test_llms.test_grpo_old_logprobs import (
    NUM_ROWS,
    _batch,
    _make_grpo,
    _record_step_gradients,
    _sampling_logps,
)


def _make_opd(**overrides: Any) -> OPD:
    agent = _make_grpo(OPD, **overrides)
    assert isinstance(agent, OPD)
    return agent


def _tied_experiences() -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Rows of :func:`_batch` with equal rewards, so every group advantage is zero."""
    ids, mask = _batch()
    return list(ids.split(1)), list(mask.split(1)), torch.ones(NUM_ROWS)


def _mean_action_log_prob(agent: GRPO) -> float:
    return float(torch.cat(_sampling_logps(agent)).mean())


class TestOPDInit:
    def test_defaults_sample_the_unmodified_policy_without_a_reference(self) -> None:
        agent = _make_opd()

        assert (agent.temperature, agent.top_k, agent.beta) == (1.0, 0, 0.0)
        assert (agent.teacher_coef, agent.teacher_gap_clip) == (1.0, 10.0)

    def test_signature_lists_the_teacher_settings_for_cloning(self) -> None:
        parameters = inspect.signature(OPD.__init__).parameters

        assert parameters["teacher_coef"].default == 1.0
        assert parameters["teacher_gap_clip"].default == 10.0
        assert parameters["temperature"].default == 1.0

    @pytest.mark.parametrize(
        ("setting", "match"),
        [
            ({"teacher_coef": 0.0}, "teacher_coef must be > 0"),
            ({"teacher_gap_clip": 0.0}, "teacher_gap_clip must be > 0"),
            ({"whiten_advantages": True}, "whiten_advantages and filter_zero_adv"),
            ({"filter_zero_adv": True}, "whiten_advantages and filter_zero_adv"),
            ({"temperature": 0.9}, "unmodified policy"),
            ({"top_k": 50}, "unmodified policy"),
            ({"top_p": 0.95}, "unmodified policy"),
            ({"min_p": 0.1}, "unmodified policy"),
        ],
    )
    def test_rejects_settings_that_change_the_scored_distribution(
        self, setting: dict[str, Any], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            _make_opd(**setting)

    def test_allows_a_group_of_one_where_grpo_does_not(self) -> None:
        agent = _make_opd(group_size=1, mini_batch_size=2)

        assert agent.group_size == 1
        with pytest.raises(ValueError, match="group_size must be >= 2"):
            _make_grpo(group_size=1, mini_batch_size=2)


class TestOPDLearn:
    def test_teacher_above_the_student_raises_sampled_log_probs(self) -> None:
        # Arrange: tied rewards, so the teacher term is the only advantage.
        agent = _make_opd()
        sampling_logps = _sampling_logps(agent)
        before = _mean_action_log_prob(agent)

        # Act
        metrics = agent.learn(
            _tied_experiences(),
            sampling_logps=sampling_logps,
            teacher_logps=[row + 0.5 for row in sampling_logps],
        )

        # Assert
        assert metrics["teacher_reverse_kl"] == pytest.approx(-0.5, abs=1e-6)
        assert metrics["adv_mean"] == pytest.approx(0.5, abs=1e-6)
        assert _mean_action_log_prob(agent) > before

    def test_teacher_below_the_student_lowers_sampled_log_probs(self) -> None:
        agent = _make_opd()
        sampling_logps = _sampling_logps(agent)
        before = _mean_action_log_prob(agent)

        metrics = agent.learn(
            _tied_experiences(),
            sampling_logps=sampling_logps,
            teacher_logps=[row - 0.5 for row in sampling_logps],
        )

        assert metrics["teacher_reverse_kl"] == pytest.approx(0.5, abs=1e-6)
        assert _mean_action_log_prob(agent) < before

    def test_teacher_matching_the_student_gives_no_gradient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = _make_opd()
        sampling_logps = _sampling_logps(agent)
        steps = _record_step_gradients(agent, monkeypatch)

        # Act
        metrics = agent.learn(
            _tied_experiences(),
            sampling_logps=sampling_logps,
            teacher_logps=[row.clone() for row in sampling_logps],
        )

        # Assert
        assert metrics["teacher_reverse_kl"] == pytest.approx(0.0, abs=1e-6)
        assert steps
        assert all(not grad.any() for step in steps for grad in step.values())

    def test_task_advantage_and_teacher_term_add(self) -> None:
        # Arrange: rewards split each group into +0.5 / -0.5 (mean_only).
        agent = _make_opd(teacher_coef=2.0)
        sampling_logps = _sampling_logps(agent)
        ids, mask = _batch()
        experiences = (
            list(ids.split(1)),
            list(mask.split(1)),
            torch.tensor([1.0, 0.0, 0.0, 1.0]),
        )

        # Act
        metrics = agent.learn(
            experiences,
            sampling_logps=sampling_logps,
            teacher_logps=[row + 0.25 for row in sampling_logps],
        )

        # Assert: per-token advantages are +-0.5 plus 2 * 0.25.
        assert metrics["adv_max"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["adv_min"] == pytest.approx(0.0, abs=1e-6)

    def test_the_gap_clip_bounds_each_token_term(self) -> None:
        # Arrange: the teacher sits 20 nats above the student; the clip is 10.
        agent = _make_opd()
        sampling_logps = _sampling_logps(agent)

        # Act
        metrics = agent.learn(
            _tied_experiences(),
            sampling_logps=sampling_logps,
            teacher_logps=[row + 20.0 for row in sampling_logps],
        )

        # Assert: the advantage is clipped; the reported reverse KL is not.
        assert metrics["adv_max"] == pytest.approx(10.0, abs=1e-5)
        assert metrics["teacher_reverse_kl"] == pytest.approx(-20.0, abs=1e-5)
        assert metrics["teacher_gap_clip_frac"] == pytest.approx(1.0)

    def test_a_group_of_one_trains_on_the_teacher_term(self) -> None:
        agent = _make_opd(group_size=1, mini_batch_size=2)
        sampling_logps = _sampling_logps(agent)
        before = _mean_action_log_prob(agent)

        metrics = agent.learn(
            _tied_experiences(),
            sampling_logps=sampling_logps,
            teacher_logps=[row + 0.5 for row in sampling_logps],
        )

        assert metrics["adv_mean"] == pytest.approx(0.5, abs=1e-6)
        assert _mean_action_log_prob(agent) > before

    def test_extra_token_advantages_add_to_the_teacher_term(self) -> None:
        agent = _make_opd()
        sampling_logps = _sampling_logps(agent)

        metrics = agent.learn(
            _tied_experiences(),
            sampling_logps=sampling_logps,
            token_advantages=[torch.full_like(row, 0.25) for row in sampling_logps],
            teacher_logps=[row + 0.5 for row in sampling_logps],
        )

        assert metrics["adv_mean"] == pytest.approx(0.75, abs=1e-6)

    def test_non_finite_teacher_log_probs_are_rejected(self) -> None:
        agent = _make_opd()
        sampling_logps = _sampling_logps(agent)
        teacher_logps = [row.clone() for row in sampling_logps]
        teacher_logps[2][0] = float("-inf")

        with pytest.raises(ValueError, match="not finite on row 2"):
            agent.learn(
                _tied_experiences(),
                sampling_logps=sampling_logps,
                teacher_logps=teacher_logps,
            )

    def test_rows_without_teacher_log_probs_are_rejected(self) -> None:
        agent = _make_opd()
        sampling_logps = _sampling_logps(agent)

        with pytest.raises(
            ValueError, match="needs teacher_logps on every action token"
        ):
            agent.learn(
                _tied_experiences(),
                sampling_logps=sampling_logps,
                teacher_logps=[sampling_logps[0], None, None, None],
            )

    def test_fewer_teacher_rows_than_sampled_rows_are_rejected(self) -> None:
        agent = _make_opd()
        sampling_logps = _sampling_logps(agent)

        with pytest.raises(ValueError, match="got 1 and 4 rows"):
            agent.learn(
                _tied_experiences(),
                sampling_logps=sampling_logps,
                teacher_logps=sampling_logps[:1],
            )

    def test_a_row_with_miscounted_teacher_log_probs_is_rejected(self) -> None:
        agent = _make_opd()
        sampling_logps = _sampling_logps(agent)
        teacher_logps = [row.clone() for row in sampling_logps]
        teacher_logps[1] = teacher_logps[1][:1]

        with pytest.raises(ValueError, match="Row 1 has 1 teacher_logps"):
            agent.learn(
                _tied_experiences(),
                sampling_logps=sampling_logps,
                teacher_logps=teacher_logps,
            )


class TestOPDResponseMetrics:
    def test_learn_logs_the_mean_action_tokens_per_row(self) -> None:
        # Arrange
        agent = _make_opd()
        sampling_logps = _sampling_logps(agent)
        _, mask = _batch()

        # Act
        metrics = agent.learn(
            _tied_experiences(),
            sampling_logps=sampling_logps,
            teacher_logps=[row.clone() for row in sampling_logps],
        )

        # Assert
        assert metrics["response_length"] == pytest.approx(float(mask.sum()) / NUM_ROWS)
        assert 0.0 <= metrics["response_repeat_4gram_frac"] <= 1.0

    def test_a_response_that_loops_counts_its_repeated_4grams(self) -> None:
        # Arrange: the response repeats 5 6 7 8; its 5 4-grams hold 4 distinct.
        looping = torch.tensor([[0, 5, 6, 7, 8, 5, 6, 7, 8]])
        short = torch.tensor([[0, 5, 6, 7]])

        # Act
        frac = _repeated_ngram_frac(
            [looping, short],
            [torch.ones(1, 8, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool)],
        )

        # Assert: the 3-token response has no 4-gram and is left out.
        assert frac == pytest.approx(0.2)

    def test_prompt_tokens_are_not_counted(self) -> None:
        ids = torch.tensor([[5, 6, 7, 8, 5, 6, 7, 8, 1, 2, 3, 4]])
        mask = torch.tensor([[False] * 7 + [True] * 4])

        assert _repeated_ngram_frac([ids], [mask]) == pytest.approx(0.0)

    def test_learn_warns_once_when_most_4grams_repeat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: the tiny model samples 4 tokens, too few to loop, so the
        # repetition share is set directly.
        agent = _make_opd()
        sampling_logps = _sampling_logps(agent)
        monkeypatch.setattr(opd_module, "_repeated_ngram_frac", lambda *_: 0.75)

        def learn() -> dict[str, Any]:
            return agent.learn(
                _tied_experiences(),
                sampling_logps=sampling_logps,
                teacher_logps=[row.clone() for row in sampling_logps],
            )

        # Act / Assert
        with pytest.warns(UserWarning, match="75% of OPD responses' 4-grams repeat"):
            metrics = learn()
        assert metrics["response_repeat_4gram_frac"] == 0.75
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            learn()

    def test_learn_does_not_warn_on_varied_responses(self) -> None:
        agent = _make_opd()
        sampling_logps = _sampling_logps(agent)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            agent.learn(
                _tied_experiences(),
                sampling_logps=sampling_logps,
                teacher_logps=[row.clone() for row in sampling_logps],
            )


def _adapter_weights(agent: GRPO) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in agent.actor.named_parameters()
        if "lora_" in name
    }


class TestOPDCheckpoint:
    def test_load_restores_the_trained_adapter_and_teacher_settings(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        trained = _make_opd(teacher_coef=0.5, teacher_gap_clip=4.0)
        sampling_logps = _sampling_logps(trained)
        trained.learn(
            _tied_experiences(),
            sampling_logps=sampling_logps,
            teacher_logps=[row + 0.5 for row in sampling_logps],
        )
        trained.save_checkpoint(str(tmp_path))
        restored = _make_opd()
        assert not all(
            torch.equal(param, _adapter_weights(trained)[name])
            for name, param in _adapter_weights(restored).items()
        )

        # Act
        restored.load_checkpoint(str(tmp_path))

        # Assert
        expected = _adapter_weights(trained)
        assert expected
        for name, param in _adapter_weights(restored).items():
            assert torch.equal(param, expected[name]), name
        assert (restored.teacher_coef, restored.teacher_gap_clip) == (0.5, 4.0)
        assert isinstance(restored, OPD)

    def test_a_restored_agent_keeps_training(self, tmp_path: Path) -> None:
        # Arrange
        trained = _make_opd()
        trained.save_checkpoint(str(tmp_path))
        restored = _make_opd()
        restored.load_checkpoint(str(tmp_path))
        sampling_logps = _sampling_logps(restored)
        before = _mean_action_log_prob(restored)

        # Act
        metrics = restored.learn(
            _tied_experiences(),
            sampling_logps=sampling_logps,
            teacher_logps=[row + 0.5 for row in sampling_logps],
        )

        # Assert
        assert metrics["teacher_reverse_kl"] == pytest.approx(-0.5, abs=1e-6)
        assert _mean_action_log_prob(restored) > before

    def test_clone_keeps_the_teacher_settings_and_adapter(self) -> None:
        # Arrange
        agent = _make_opd(teacher_coef=0.5, teacher_gap_clip=None)

        # Act
        clone = agent.clone()

        # Assert
        assert isinstance(clone, OPD)
        assert (clone.teacher_coef, clone.teacher_gap_clip) == (0.5, None)
        expected = _adapter_weights(agent)
        for name, param in _adapter_weights(clone).items():
            assert torch.equal(param, expected[name]), name


class TestGRPOLearnTokenAdvantages:
    def test_terms_add_to_the_group_advantages(self) -> None:
        # Arrange
        agent = _make_grpo()
        sampling_logps = _sampling_logps(agent)

        # Act
        metrics = agent.learn(
            _tied_experiences(),
            sampling_logps=sampling_logps,
            token_advantages=[torch.full_like(row, 0.75) for row in sampling_logps],
        )

        # Assert
        assert metrics["adv_mean"] == pytest.approx(0.75, abs=1e-6)

    def test_a_row_with_miscounted_terms_gets_no_term(self) -> None:
        # Arrange
        agent = _make_grpo()
        sampling_logps = _sampling_logps(agent)
        terms = [torch.full_like(row, 0.75) for row in sampling_logps]
        terms[1] = terms[1][:1]
        token_count = sum(row.numel() for row in sampling_logps)
        expected = 0.75 * (token_count - sampling_logps[1].numel()) / token_count

        # Act
        with pytest.warns(UserWarning, match=r"Rows \[1\] of token_advantages"):
            metrics = agent.learn(
                _tied_experiences(),
                sampling_logps=sampling_logps,
                token_advantages=terms,
            )

        # Assert
        assert metrics["adv_mean"] == pytest.approx(expected, abs=1e-6)

    def test_an_empty_list_of_terms_is_rejected(self) -> None:
        agent = _make_grpo()

        with pytest.raises(ValueError, match="must name at least one row"):
            agent.learn(
                _tied_experiences(),
                sampling_logps=_sampling_logps(agent),
                token_advantages=[],
            )

    def test_filter_zero_adv_is_rejected(self) -> None:
        agent = _make_grpo(filter_zero_adv=True)
        sampling_logps = _sampling_logps(agent)

        with pytest.raises(ValueError, match="filter_zero_adv drops rows"):
            agent.learn(
                _tied_experiences(),
                sampling_logps=sampling_logps,
                token_advantages=[torch.full_like(row, 0.75) for row in sampling_logps],
            )
