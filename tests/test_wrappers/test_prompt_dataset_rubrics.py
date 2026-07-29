# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Focused coverage for PromptDatasetEnv and reward_fn_to_rubric."""

from __future__ import annotations

import pytest
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import Sequential

from agilerl.llm_envs.openenv import LocalEnvClient, TextAction
from agilerl.llm_envs.prompt_dataset import PromptDatasetEnv
from agilerl.llm_envs.rubrics import reward_fn_to_rubric


class _Leaf(Rubric):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, action, observation) -> float:
        del action, observation
        return self.value


class _Parent(Rubric):
    """OpenEnv-style composition: children register via attribute assignment."""

    def __init__(self, **children: Rubric) -> None:
        super().__init__()
        for name, child in children.items():
            setattr(self, name, child)

    def forward(self, action, observation) -> float:
        return float(sum(child(action, observation) for child in self.children()))


def test_reward_fn_to_rubric_positional_call() -> None:
    seen: list[tuple] = []

    def fn(completion, answer, question):
        seen.append((completion, answer, question))
        return 0.5

    rubric = reward_fn_to_rubric(fn)
    rows = [{"question": "q", "answer": "a"}]
    env = PromptDatasetEnv(rows, rubric=rubric)
    env.reset(row_index=0)
    obs = env.step(TextAction(message="done"))
    assert obs.reward == 0.5
    assert seen == [("done", "a", "q")]
    assert env.rubric_components == ("fn",)


def test_prompt_dataset_rejects_callable_rubric() -> None:
    with pytest.raises(TypeError, match="reward_fn_to_rubric"):
        PromptDatasetEnv(
            [{"question": "q", "answer": "a"}],
            rubric=lambda c, a, q: 1.0,  # type: ignore[arg-type]
        )


def test_reset_hides_labels() -> None:
    env = PromptDatasetEnv(
        [{"question": "q", "answer": "secret"}],
        rubric=reward_fn_to_rubric(lambda c, a, q: 1.0),
    )
    obs = env.reset(row_index=0)
    assert obs.answer is None
    assert obs.question is None
    assert obs.prompt == "q"


def test_terminal_obs_keeps_rendered_prompt() -> None:
    """Terminal step keeps the rendered prompt (may differ from ``question``)."""
    seen: list[str] = []

    class _PromptRubric(Rubric):
        def forward(self, action, observation) -> float:
            del action
            seen.append(observation.prompt)
            return 1.0 if observation.prompt.startswith("Ask:") else 0.0

    env = PromptDatasetEnv(
        [{"question": "q", "answer": "a"}],
        rubric=_PromptRubric(),
        prompt_builder=lambda row: f"Ask: {row['question']}",
    )
    assert env.reset(row_index=0).prompt == "Ask: q"
    terminal = env.step(TextAction(message="x"))
    assert terminal.prompt == "Ask: q"
    assert terminal.question == "q"
    assert terminal.answer == "a"
    assert terminal.reward == 1.0
    assert seen == ["Ask: q"]


def test_openenv_children_report_components() -> None:
    env = PromptDatasetEnv(
        [{"question": "q", "answer": "a"}],
        rubric=_Parent(fmt=_Leaf(1.0), correct=_Leaf(1.0)),
    )
    env.reset(row_index=0)
    obs = env.step(TextAction(message="x"))
    assert obs.reward == 2.0
    assert obs.metadata["rubric_scores"] == {"fmt": 1.0, "correct": 1.0}
    assert env.rubric_components == ("fmt", "correct")


def test_each_env_owns_its_rubric_hooks() -> None:
    """Each env should get its own rubric tree (one hook per leaf per env)."""
    rows = [{"question": str(i), "answer": str(i)} for i in range(4)]
    envs = [PromptDatasetEnv(rows, rubric=_Parent(a=_Leaf(1.0))) for _ in range(3)]
    assert all(env.rubric_components == ("a",) for env in envs)
    for env in envs:
        leaf = next(env.rubric.children())
        assert len(leaf._forward_hooks) == 1


def test_short_circuit_omits_skipped_child() -> None:
    class Zero(Rubric):
        def forward(self, action, observation) -> float:
            del action, observation
            return 0.0

    class Never(Rubric):
        def forward(self, action, observation) -> float:
            del action, observation
            msg = "should be short-circuited"
            raise AssertionError(msg)

    env = PromptDatasetEnv(
        [{"question": "q", "answer": "a"}],
        rubric=Sequential(Zero(), Never()),
    )
    env.reset(row_index=0)
    obs = env.step(TextAction(message="x"))
    assert obs.reward == 0.0
    scores = obs.metadata.get("rubric_scores") or {}
    assert "rubric_0" in scores
    assert "rubric_1" not in scores


def test_local_env_client_forwards_rubric_scores() -> None:
    """``LocalEnvClient`` surfaces leaf ``rubric_scores`` in step info."""
    env = PromptDatasetEnv(
        [{"question": "q", "answer": "a"}],
        rubric=_Parent(fmt=_Leaf(1.0), correct=_Leaf(1.0)),
    )
    client = LocalEnvClient(env)
    assert client.rubric_components == ("fmt", "correct")
    client.reset(row_index=0)
    _prompt, _reward, _terminated, _truncated, info = client.step("x")
    assert info["rubric_scores"] == {"fmt": 1.0, "correct": 1.0}


def test_local_env_client_does_not_double_wrap_prompt_dataset() -> None:
    """OpenEnv ``Environment`` instances are stored as-is on the client."""
    world = PromptDatasetEnv(
        [{"question": "q", "answer": "a"}],
        rubric=reward_fn_to_rubric(lambda c, a, q: 1.0),
    )
    client = LocalEnvClient(world)
    assert client._env is world
