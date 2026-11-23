# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The GSM8K tutorial's reward.py scores turns with and without reasoning."""

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


def score(completion: str, answer: str = GOLD) -> float:
    rubric = reward_fn_to_rubric(load_reward().reward_fn)
    action = TurnAction(message=completion, completion=completion)
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
        "completion",
        [
            # Qwen3.5 thinking on: the prompt holds the opening <think>.
            "48 / 2 = 24, and 48 + 24 = 72.\n</think>\n\n<answer>72</answer>",
            # A model taught the tags by its prompt writes both.
            "<think>48 / 2 = 24, and 48 + 24 = 72.</think>\n<answer>72</answer>",
            # Thinking off: the model reasons in plain text.
            "48 / 2 = 24, and 48 + 24 = 72.\n<answer>72</answer>",
        ],
        ids=["thinking-on", "think-tags", "thinking-off"],
    )
    def test_a_correct_tagged_answer_scores_full_marks(self, completion):
        assert score(completion) == 2.0

    def test_an_answer_inside_the_reasoning_does_not_count(self):
        assert score("<answer>72</answer>\n</think>\n\n<answer>70</answer>") == 1.0

    def test_text_after_the_answer_loses_the_format_reward(self):
        assert score("<answer>72</answer>\nHope that helps!") == 1.0

    def test_an_untagged_answer_scores_zero(self):
        assert score("She sold 72 clips.") == 0.0

    def test_unfinished_reasoning_scores_zero(self):
        assert score("48 / 2 = 24, so <answer>72</answer> maybe</think>") == 0.0

    def test_thousands_separators_match(self):
        assert score("<answer>1,200</answer>", answer="... #### 1200") == 2.0
