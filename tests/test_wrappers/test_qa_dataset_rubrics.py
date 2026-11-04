# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Focused coverage for QADatasetEnv and the rubric helpers."""

from __future__ import annotations

import numpy as np
import pytest
from openenv.core.rubrics.base import Rubric

from agilerl.llm_envs.openenv import InProcessEnvClient
from agilerl.llm_envs.openenv_server import TextAction
from agilerl.llm_envs.qa_dataset import QADatasetEnv, _json_safe
from agilerl.llm_envs.rubrics import (
    EnvReward,
    TurnAction,
    TurnObservation,
    register_component_hooks,
    reward_fn_to_rubric,
)


def _turn(env_reward: float = 0.0) -> tuple[TurnAction, TurnObservation]:
    return (
        TurnAction(message="done", completion="<think>r</think>done"),
        TurnObservation(
            env_reward=env_reward,
            env_ran=True,
            done=True,
            turn_index=0,
            prompt="P:q",
            question="q",
            answer="a",
        ),
    )


def test_reset_hides_labels() -> None:
    env = QADatasetEnv([{"question": "q", "answer": "secret"}])
    obs = env.reset(row_index=0)
    assert obs.answer is None
    assert obs.question is None
    assert obs.prompt == "q"


def test_reset_seed_pins_row_when_row_index_is_absent() -> None:
    rows = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(4)]
    env = QADatasetEnv(rows)

    first = env.reset(seed=6)
    second = env.reset(seed=6)

    assert first.prompt == "q2"
    assert second.prompt == "q2"


def test_reset_row_index_wins_over_seed() -> None:
    rows = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(4)]
    env = QADatasetEnv(rows)

    obs = env.reset(seed=6, row_index=0)

    assert obs.prompt == "q0"


def test_reset_row_index_out_of_range_raises() -> None:
    rows = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(2)]
    env = QADatasetEnv(rows)

    with pytest.raises(IndexError, match="out of range"):
        env.reset(row_index=2)


def test_eval_row_index_out_of_range_does_not_wrap() -> None:
    train = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(8)]
    test = [{"question": "tq", "answer": "ta"}]
    env = QADatasetEnv(train, test_dataset=test)

    with pytest.raises(IndexError, match="out of range"):
        env.reset(row_index=1, evaluation=True)


def test_terminal_obs_carries_labels_and_no_reward() -> None:
    """Terminal step keeps the rendered prompt and adds labels; the env does not score."""
    env = QADatasetEnv(
        [{"question": "q", "answer": "a"}],
        prompt_builder=lambda row: f"Ask: {row['question']}",
    )
    assert env.reset(row_index=0).prompt == "Ask: q"

    terminal = env.step(TextAction(message="x"))

    assert terminal.prompt == "Ask: q"
    assert terminal.question == "q"
    assert terminal.answer == "a"
    assert terminal.reward == 0.0
    assert terminal.done is True


def test_local_env_client_does_not_double_wrap_qa_dataset() -> None:
    """OpenEnv ``Environment`` instances are stored as-is on the client."""
    world = QADatasetEnv([{"question": "q", "answer": "a"}])
    client = InProcessEnvClient(world)
    assert client._backend is world


def test_json_safe_coerces_numpy_and_nested_containers() -> None:
    assert _json_safe(np.int64(3)) == 3
    assert _json_safe(np.array([1, 2])) == [1, 2]
    coerced = _json_safe({"k": np.float32(1.5)})
    assert isinstance(coerced, dict)
    assert coerced["k"] == pytest.approx(1.5)
    assert _json_safe((np.int64(1), "x")) == [1, "x"]


def test_json_safe_warns_on_unrecognised_type() -> None:
    with pytest.warns(UserWarning, match="not coerced"):
        assert _json_safe(object()) is not None


def test_qa_dataset_empty_rows_raises() -> None:
    env = QADatasetEnv([])
    with pytest.raises(RuntimeError, match="no rows"):
        env.reset()


class TestRewardFnToRubric:
    def test_calls_fn_with_message_answer_question(self) -> None:
        seen: list[tuple] = []

        def fn(completion, answer, question):
            seen.append((completion, answer, question))
            return 0.5

        action, observation = _turn()

        score = reward_fn_to_rubric(fn)(action, observation)

        assert score == 0.5
        assert seen == [("done", "a", "q")]

    def test_binds_arguments_by_name(self) -> None:
        seen: list[tuple] = []

        def fn(question, answer, completion):
            seen.append((question, answer, completion))
            return 0.25

        action, observation = _turn()

        score = reward_fn_to_rubric(fn)(action, observation)

        assert score == 0.25
        assert seen == [("q", "a", "done")]

    def test_names_the_component_after_the_fn(self) -> None:
        def accuracy(completion, answer, question):
            return 1.0

        assert register_component_hooks(reward_fn_to_rubric(accuracy)) == ("accuracy",)


class TestEnvReward:
    def test_returns_the_env_reward(self) -> None:
        action, observation = _turn(env_reward=0.75)

        assert EnvReward()(action, observation) == 0.75

    def test_reports_as_the_env_component(self) -> None:
        rubric = EnvReward()
        register_component_hooks(rubric)
        action, observation = _turn(env_reward=0.75)

        rubric(action, observation)

        assert observation.metadata["rubric_scores"] == {"env": 0.75}


def test_register_component_hooks_names_a_bare_leaf_from_its_class() -> None:
    class _OddName(Rubric):
        def forward(self, action, observation) -> float:
            del action, observation
            return 1.0

    assert register_component_hooks(_OddName()) == ("odd_name",)


def test_shared_rubric_scores_each_observation_once() -> None:
    """Repeat registration still stamps each observation's one score."""
    rubric = reward_fn_to_rubric(lambda completion, answer, question: 0.5, "acc")
    for _ in range(3):
        register_component_hooks(rubric)

    for _ in range(3):
        action, observation = _turn()
        rubric(action, observation)
        assert observation.metadata["rubric_scores"] == {"acc": 0.5}
