# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The GSM8K tutorial's reward.py scores Qwen3.5 turns with thinking on and off."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from agilerl.llm_envs.rubrics import TurnAction, TurnObservation, reward_fn_to_rubric

REWARD_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "_static"
    / "examples"
    / "gsm8k-grpo"
    / "reward.py"
)
GOLD = "Natalia sold 48/2 = 24 clips in May.\nAltogether she sold 48+24 = 72 clips.\n#### 72"


def load_reward() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gsm8k_reward", REWARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(completion: str, response: str, answer: str = GOLD) -> float:
    rubric = reward_fn_to_rubric(load_reward().reward_fn)
    action = TurnAction(message=response, completion=completion, response=response)
    observation = TurnObservation(
        env_reward=0.0,
        env_ran=True,
        done=True,
        turn_index=0,
        prompt="",
        question="How many clips?",
        answer=answer,
    )
    return rubric(action, observation)


class TestGsm8kTutorialReward:
    @pytest.mark.parametrize(
        ("completion", "response"),
        [
            # Thinking on: the prompt holds the opening <think>.
            (
                "48 / 2 = 24, and 48 + 24 = 72.\n</think>\n\n<answer>72</answer>",
                "<answer>72</answer>",
            ),
            # Thinking off: the template closed the block; the model reasons in plain text.
            (
                "48 / 2 = 24, and 48 + 24 = 72.\n<answer>72</answer>",
                "48 / 2 = 24, and 48 + 24 = 72.\n<answer>72</answer>",
            ),
        ],
        ids=["thinking-on", "thinking-off"],
    )
    def test_a_correct_tagged_answer_scores_full_marks(self, completion, response):
        assert score(completion, response) == 2.0

    def test_an_answer_inside_the_reasoning_does_not_count(self):
        completion = "<answer>72</answer>\n</think>\n\n<answer>70</answer>"

        assert score(completion, "<answer>70</answer>") == 1.0

    def test_text_after_the_answer_loses_the_format_reward(self):
        response = "<answer>72</answer>\nHope that helps!"

        assert score(response, response) == 1.0

    def test_an_untagged_answer_scores_zero(self):
        response = "She sold 72 clips."

        assert score(response, response) == 0.0

    def test_thousands_separators_match(self):
        response = "<answer>1,200</answer>"

        assert score(response, response, answer="... #### 1200") == 2.0

    def test_without_response_it_reads_the_completion_after_the_reasoning(self):
        reward_fn = load_reward().reward_fn

        reward = reward_fn(
            completion="72?\n</think>\n\n<answer>72</answer>",
            answer=GOLD,
            question="How many clips?",
        )

        assert reward == 2.0
