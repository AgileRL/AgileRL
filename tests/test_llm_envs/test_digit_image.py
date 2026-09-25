# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Digit image env behavior."""

from __future__ import annotations

import pytest
from PIL import Image

from agilerl.llm_envs.digit_image import make


class TestDigitImageEnv:
    def test_reset_returns_text_and_pil_image(self) -> None:
        # Arrange
        env = make()

        # Act
        obs, info = env.reset(seed=0)

        # Assert
        assert isinstance(obs, dict)
        assert "text" in obs
        assert "image" in obs
        assert isinstance(obs["image"], Image.Image)
        assert info == {}

    def test_step_rewards_matching_digit_and_ends_episode(self) -> None:
        # Arrange
        env = make()
        env.reset(seed=1)
        digit_char = str(env._digit)

        # Act
        _obs, reward, terminated, truncated, _info = env.step(digit_char)

        # Assert
        assert reward == 1.0
        assert terminated is True
        assert truncated is False

    def test_step_before_reset_raises(self) -> None:
        env = make()

        with pytest.raises(RuntimeError, match=r"step\(\) before reset\(\)"):
            env.step("1")

    def test_step_zero_reward_for_wrong_answer(self) -> None:
        # Arrange
        env = make()
        env.reset(seed=2)
        wrong = "not-a-digit"

        # Act
        _obs, reward, terminated, truncated, _info = env.step(wrong)

        # Assert
        assert reward == 0.0
        assert terminated is True
        assert truncated is False
