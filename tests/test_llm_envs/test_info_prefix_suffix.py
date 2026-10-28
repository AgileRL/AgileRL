# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Env ``info`` prefix/suffix merged into the prompt text.

Game envs in the gem convention return instructions as the observation and the
task body as ``info["suffix"]``; without the merge the model never sees the board.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from agilerl.llm_envs import RolloutHarness
from agilerl.llm_envs.observation import apply_info_prefix_suffix
from tests.helpers.rollout_doubles import MiniTokenizer


class TestApplyInfoPrefixSuffix:
    def test_appends_the_suffix(self) -> None:
        assert (
            apply_info_prefix_suffix("instructions", {"suffix": "board"})
            == "instructionsboard"
        )

    def test_prepends_the_prefix(self) -> None:
        assert apply_info_prefix_suffix("body", {"prefix": "head"}) == "headbody"

    def test_applies_both(self) -> None:
        assert (
            apply_info_prefix_suffix("body", {"prefix": "head", "suffix": "tail"})
            == "headbodytail"
        )

    def test_leaves_text_without_affixes_untouched(self) -> None:
        assert apply_info_prefix_suffix("body", {"other": 1}) == "body"

    def test_ignores_non_string_affixes(self) -> None:
        assert apply_info_prefix_suffix("body", {"prefix": None, "suffix": 7}) == "body"


class _AffixEnvClient:
    """Env client double returning gem-style info suffixes."""

    def reset(
        self, seed: int | None = None, *, row_index: int | None = None
    ) -> tuple[Any, dict[str, Any]]:
        del seed, row_index
        return "instructions", {"suffix": "board"}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        del action
        return "moved", 1.0, False, False, {"suffix": "board2"}

    def close(self) -> None:
        return None

    @property
    def dataset_size(self) -> int:
        return 0

    @property
    def tools(self) -> list[Any]:
        return []

    @property
    def takes_tool_calls(self) -> bool:
        return False

    @property
    def answer_tool(self) -> str | None:
        return None

    @property
    def rubric_components(self) -> tuple[str, ...]:
        return ()

    @contextmanager
    def eval_mode(self) -> Iterator[None]:
        yield


class TestHarnessInfoAffixes:
    def test_reset_prompt_carries_the_suffix(self) -> None:
        harness = RolloutHarness(
            _AffixEnvClient(),
            MiniTokenizer(),
            max_turns=3,
            apply_chat_template=False,
        )

        obs_text, _image, _info = harness._reset_fetch()

        assert obs_text == "instructionsboard"

    def test_step_feedback_carries_the_suffix(self) -> None:
        harness = RolloutHarness(
            _AffixEnvClient(),
            MiniTokenizer(),
            max_turns=3,
            apply_chat_template=False,
        )
        harness.reset()

        result = harness._step_env("go")

        assert result.messages == [{"role": "user", "content": "movedboard2"}]
