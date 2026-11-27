# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the local reward file check that runs before Arena prices a run."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agilerl.exceptions import ArenaValidationError
from agilerl.models.env import LLMEnvSpec
from agilerl.reward_check import check_reward_file

GOOD_REWARD = """\
def reward_fn(completion: str, answer: str, question: str) -> float:
    return 1.0 if completion.strip() == answer.strip() else 0.0
"""


def dataset_rollout() -> LLMEnvSpec:
    return LLMEnvSpec(
        env_type="rollout",
        dataset="gsm8k",
        rubric_file_path="reward.py",
        prompt_template={"user_0": "{question}"},
    )


def write_reward(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "reward.py"
    path.write_text(source)
    return path


class TestCheckRewardFile:
    def test_accepts_a_reward_that_returns_a_float(self, tmp_path: Path) -> None:
        # Arrange
        path = write_reward(tmp_path, GOOD_REWARD)

        # Act / Assert
        check_reward_file(path, dataset_rollout())

    @pytest.mark.parametrize("value", ["1", "numpy.float32(0.5)"])
    def test_accepts_a_number_the_trainer_converts(
        self, tmp_path: Path, value: str
    ) -> None:
        # Arrange
        source = (
            "import numpy\n"
            "def reward_fn(completion: str, answer: str, question: str) -> float:\n"
            f"    return {value}\n"
        )
        path = write_reward(tmp_path, source)

        # Act / Assert
        check_reward_file(path, dataset_rollout())

    def test_accepts_reward_bytes(self) -> None:
        check_reward_file(GOOD_REWARD.encode(), dataset_rollout())

    def test_names_the_line_of_a_syntax_error(self, tmp_path: Path) -> None:
        # Arrange
        path = write_reward(tmp_path, "def reward_fn(completion, answer, question)\n")

        # Act / Assert
        with pytest.raises(
            ArenaValidationError,
            match=rf"{re.escape(str(path))}, line 1: SyntaxError: expected ':'",
        ):
            check_reward_file(path, dataset_rollout())

    def test_names_the_line_of_an_undefined_name_at_import(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        path = write_reward(tmp_path, "SCALE = 1.0\nOFFSET = missing_offset\n")

        # Act / Assert
        with pytest.raises(
            ArenaValidationError,
            match=rf"{re.escape(str(path))}, line 2: NameError: name 'missing_offset' is not defined",
        ):
            check_reward_file(path, dataset_rollout())

    def test_names_the_line_of_an_undefined_name_in_the_reward(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        source = (
            "def reward_fn(completion: str, answer: str, question: str) -> float:\n"
            "    score = 1.0\n"
            "    return undefined_name + score\n"
        )
        path = write_reward(tmp_path, source)

        # Act / Assert
        with pytest.raises(
            ArenaValidationError,
            match=rf"{re.escape(str(path))}, line 3: NameError: name 'undefined_name' is not defined",
        ):
            check_reward_file(path, dataset_rollout())

    @pytest.mark.parametrize(
        ("value", "kind"), [("'high'", "str"), ("None", "NoneType"), ("True", "bool")]
    )
    def test_rejects_a_reward_that_is_not_a_float(
        self, tmp_path: Path, value: str, kind: str
    ) -> None:
        # Arrange
        source = (
            "def reward_fn(completion: str, answer: str, question: str) -> float:\n"
            f"    return {value}\n"
        )
        path = write_reward(tmp_path, source)

        # Act / Assert
        with pytest.raises(
            ArenaValidationError,
            match=rf"line 1: reward_fn returned {kind} .*a reward must return a float",
        ):
            check_reward_file(path, dataset_rollout())

    def test_rejects_a_file_without_the_named_reward(self, tmp_path: Path) -> None:
        # Arrange
        path = write_reward(
            tmp_path, "def score(completion, answer, question):\n    return 0.0\n"
        )

        # Act / Assert
        with pytest.raises(ArenaValidationError, match="defines no 'reward_fn'"):
            check_reward_file(path, dataset_rollout())

    @pytest.mark.parametrize(
        "source",
        [
            "import package_only_the_trainer_has\n" + GOOD_REWARD,
            (
                "def reward_fn(completion: str, answer: str, question: str) -> float:\n"
                "    return float(answer.split('####')[1])\n"
            ),
            "def reward_fn(prompt: str, response: str) -> float:\n    return 0.0\n",
            "class reward_fn:\n    pass\n",
        ],
        ids=["missing-package", "needs-a-real-row", "other-arguments", "rubric-class"],
    )
    def test_leaves_what_a_real_row_or_the_trainer_decides(
        self, tmp_path: Path, source: str
    ) -> None:
        # Arrange
        path = write_reward(tmp_path, source)

        # Act / Assert
        check_reward_file(path, dataset_rollout())

    def test_does_not_call_the_reward_of_an_env_rollout(self, tmp_path: Path) -> None:
        # Arrange
        source = "def reward_fn(completion, answer, question):\n    return 'high'\n"
        path = write_reward(tmp_path, source)
        environment = LLMEnvSpec(
            env_type="rollout", entrypoint="envs:MyEnv", rubric_file_path="reward.py"
        )

        # Act / Assert
        check_reward_file(path, environment)
