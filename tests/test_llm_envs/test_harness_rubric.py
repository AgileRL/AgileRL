# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""A harness rubric scores every turn of a plain env, in place of the env's reward."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import Sequential, WeightedSum

from agilerl.llm_envs import RolloutHarness
from agilerl.llm_envs.rubrics import EnvReward, TurnAction, TurnObservation


class ChrTokenizer:
    """One token per character; ``</think>`` encodes as its characters."""

    pad_token_id = 0
    pad_token = "<pad>"
    unk_token_id = None

    def convert_tokens_to_ids(self, _token: str) -> None:
        return None

    def __call__(self, texts, **kwargs):
        t = torch.tensor([[ord(c) for c in texts[0]]], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def encode(self, s: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in s]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(x)) for x in ids)


class CountingEnv:
    """Text env paying ``reward`` per step; terminates after ``turns`` steps."""

    def __init__(self, reward: float = 1.0, turns: int = 1) -> None:
        self.reward = reward
        self.turns = turns
        self.steps = 0

    def reset(self, seed: int | None = None, *, row_index: int | None = None):
        self.steps = 0
        return "hello", {}

    def step(self, text: str):
        self.steps += 1
        return "next", self.reward, self.steps >= self.turns, False, {}


class Recorder(Rubric):
    component_name = "recorder"

    def __init__(self, score: float = 0.25) -> None:
        super().__init__()
        self.score = score
        self.calls: list[tuple[TurnAction, TurnObservation]] = []
        self.resets = 0

    def forward(self, action: TurnAction, observation: TurnObservation) -> float:
        self.calls.append((action, observation))
        return self.score

    def reset(self) -> None:
        self.resets += 1


class Fixed(Rubric):
    def __init__(self, name: str, score: float) -> None:
        super().__init__()
        self.component_name = name
        self.score = score

    def forward(self, _action: TurnAction, _observation: TurnObservation) -> float:
        return self.score


def make_harness(
    env: CountingEnv,
    rubric: Rubric | None,
    max_turns: int = 1,
    action_processor: Callable[[str], str] | None = None,
):
    return RolloutHarness.local(
        env,
        ChrTokenizer(),
        max_turns=max_turns,
        apply_chat_template=False,
        rubric=rubric,
        action_processor=action_processor,
    )


def step_text(
    harness: RolloutHarness, prompt: dict[str, Any], text: str
) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    generated = torch.tensor([[ord(c) for c in text]], dtype=torch.long)
    return harness.step(torch.cat([prompt["input_ids"], generated], dim=1))


class TestRolloutHarnessRubric:
    def test_rubric_score_replaces_env_reward(self) -> None:
        harness = make_harness(CountingEnv(reward=1.0), Recorder(score=0.25))
        prompt, _ = harness.reset()

        _, reward, terminated, _, _ = step_text(harness, prompt, "ans")

        assert reward == 0.25
        assert terminated is True
        assert harness.turn_rewards == [0.25]
        harness.close()

    def test_no_rubric_keeps_env_reward(self) -> None:
        harness = make_harness(CountingEnv(reward=1.0), None)
        prompt, _ = harness.reset()

        _, reward, _, _, _ = step_text(harness, prompt, "ans")

        assert reward == 1.0
        harness.close()

    def test_rubric_sees_the_env_reward_and_prompt(self) -> None:
        # Arrange
        rubric = Recorder()
        harness = make_harness(CountingEnv(reward=0.5), rubric)
        prompt, _ = harness.reset()

        # Act
        step_text(harness, prompt, "ans")

        # Assert
        (_, observation) = rubric.calls[0]
        assert observation.env_reward == 0.5
        assert observation.env_ran is True
        assert observation.done is True
        assert observation.turn_index == 0
        assert observation.prompt == "hello"
        harness.close()

    def test_message_drops_reasoning_and_completion_keeps_it(self) -> None:
        rubric = Recorder()
        harness = make_harness(CountingEnv(), rubric)
        prompt, _ = harness.reset()

        step_text(harness, prompt, "<think>r</think>ans")

        (action, _) = rubric.calls[0]
        assert action.message == "ans"
        assert action.completion == "<think>r</think>ans"
        harness.close()

    def test_message_is_the_processed_env_action(self) -> None:
        rubric = Recorder()
        harness = make_harness(
            CountingEnv(), rubric, action_processor=lambda text: f"quoted({text})"
        )
        prompt, _ = harness.reset()

        step_text(harness, prompt, "<think>r</think>ans")

        (action, _) = rubric.calls[0]
        assert action.message == "quoted(ans)"
        assert action.completion == "<think>r</think>ans"
        harness.close()

    def test_done_marks_the_last_turn_of_the_budget(self) -> None:
        rubric = Recorder()
        harness = make_harness(CountingEnv(turns=99), rubric, max_turns=2)
        prompt, _ = harness.reset()

        prompt, _, _, _, _ = step_text(harness, prompt, "a")
        _, _, _, truncated, _ = step_text(harness, prompt, "b")

        assert [obs.done for _, obs in rubric.calls] == [False, True]
        assert truncated is True
        harness.close()

    def test_env_reward_combines_with_other_components(self) -> None:
        # Arrange
        rubric = WeightedSum([EnvReward(), Fixed("fmt", 0.0)], weights=[0.5, 0.5])
        harness = make_harness(CountingEnv(reward=1.0), rubric)
        prompt, _ = harness.reset()

        # Act
        _, reward, _, _, info = step_text(harness, prompt, "ans")

        # Assert
        assert reward == 0.5
        assert info["rubric_scores"] == {"env": 1.0, "fmt": 0.0}
        assert harness.rubric_components == ("env", "fmt")
        assert harness.rubric_score_sums == {"env": 1.0, "fmt": 0.0}
        harness.close()

    def test_short_circuit_omits_the_skipped_child(self) -> None:
        class Never(Rubric):
            def forward(self, _action, _observation) -> float:
                msg = "should be short-circuited"
                raise AssertionError(msg)

        rubric = Sequential(Fixed("gate", 0.0), Never())
        harness = make_harness(CountingEnv(), rubric)
        prompt, _ = harness.reset()

        _, reward, _, _, info = step_text(harness, prompt, "ans")

        assert reward == 0.0
        assert info["rubric_scores"] == {"gate": 0.0}
        harness.close()

    def test_rubric_resets_with_each_episode(self) -> None:
        rubric = Recorder()
        harness = make_harness(CountingEnv(), rubric)

        harness.reset()
        harness.reset()

        assert rubric.resets == 2
        harness.close()

    def test_async_rubric_raises(self) -> None:
        class AsyncRubric(Rubric):
            async def forward(self, _action, _observation) -> float:
                return 1.0

        harness = make_harness(CountingEnv(), AsyncRubric())
        prompt, _ = harness.reset()

        with pytest.raises(TypeError, match="Async rubrics are not supported"):
            step_text(harness, prompt, "ans")
        harness.close()

    def test_env_reported_rubric_scores_are_summed_without_a_rubric(self) -> None:
        class SelfScoringEnv(CountingEnv):
            def step(self, text: str):
                obs, reward, terminated, truncated, _info = super().step(text)
                return (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    {"rubric_scores": {"own": 0.5}},
                )

        harness = make_harness(SelfScoringEnv(turns=2), None, max_turns=2)
        prompt, _ = harness.reset()

        prompt, _, _, _, info = step_text(harness, prompt, "a")
        step_text(harness, prompt, "b")

        assert info["rubric_scores"] == {"own": 0.5}
        assert harness.rubric_score_sums == {"own": 1.0}
        harness.close()

    def test_rejects_a_callable_rubric(self) -> None:
        with pytest.raises(TypeError, match="reward_fn_to_rubric"):
            make_harness(CountingEnv(), lambda c, a, q: 1.0)  # type: ignore[arg-type]
