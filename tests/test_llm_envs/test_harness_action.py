# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Text after the first think close, and where the system prompt sits."""

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from agilerl.llm_envs.harness import RolloutHarness, action_span
from tests import TINY_LLM_FIXTURE_PATH
from tests.helpers.rollout_doubles import FakeEnvClient, MiniTokenizer


class TestActionSpan:
    def test_keeps_a_bare_action(self) -> None:
        assert action_span("click('12')") == "click('12')"

    def test_drops_reasoning_from_a_prefilled_open_think(self) -> None:
        text = "The lamp is $5.\nSubmit it.</think>\n\nsend_msg_to_user('$5')"

        assert action_span(text) == "send_msg_to_user('$5')"

    def test_drops_a_leading_closed_think_block(self) -> None:
        text = "<think>\nlook at bid 12\n</think>\nclick('12')"

        assert action_span(text) == "click('12')"

    def test_drops_an_empty_think_block(self) -> None:
        assert action_span("<think></think>click('1996')") == "click('1996')"

    def test_keeps_the_text_after_the_first_close(self) -> None:
        text = "first</think>click('3')</think>"

        assert action_span(text) == "click('3')</think>"

    def test_passes_unclosed_reasoning_through(self) -> None:
        text = "The page lists three lamps and the cheapest is"

        assert action_span(text) == text

    def test_reasoning_with_no_action_sends_nothing(self) -> None:
        assert action_span("<think>only reasoning</think>") == ""

    def test_leaves_a_colon_answer_unchanged(self) -> None:
        assert action_span("Answer: 42") == "Answer: 42"


class RecordingClient(FakeEnvClient):
    """Env client that keeps each action it is stepped with."""

    def __init__(self) -> None:
        super().__init__()
        self.actions: list[str] = []

    def step(self, action: object) -> tuple[str, float, bool, bool, dict[str, object]]:
        self.actions.append(str(action))
        return super().step(action)


class RecordingTokenizer(MiniTokenizer):
    """Decodes every generation as ``text`` and records each prompt it tokenizes."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def decode(self, ids: object, **_: object) -> str:
        del ids
        return self.text

    def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
        self.prompts.append(texts[0])
        ids = torch.tensor([[1]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def rewrite_action(text: str) -> str:
    return f"quoted({text})"


def restarted_prompt(action: str, *, processor: object = None) -> tuple[list[str], str]:
    """Step one action and return the env's actions and the restarted prompt."""
    client = RecordingClient()
    tokenizer = RecordingTokenizer(action)
    harness = RolloutHarness(
        client,
        tokenizer,
        max_turns=2,
        apply_chat_template=False,
        segment_prompt_tokens=2,
        action_processor=processor,
    )
    harness.reset()
    harness.step(torch.tensor([[1, 9]], dtype=torch.long))
    return client.actions, tokenizer.prompts[-1]


class TestRolloutHarnessActionProcessor:
    def test_default_sends_the_action_text_unchanged(self) -> None:
        # Act
        actions, prompt = restarted_prompt("Answer: 42")

        # Assert
        assert actions == ["Answer: 42"]
        assert "1. Answer: 42" in prompt

    def test_processor_rewrites_the_env_action_and_the_restart_history(self) -> None:
        # Act
        actions, prompt = restarted_prompt("fill: hello", processor=rewrite_action)

        # Assert
        assert actions == ["quoted(fill: hello)"]
        assert "1. quoted(fill: hello)" in prompt


class ObservationClient(FakeEnvClient):
    """Env client whose reset returns ``observation``."""

    def __init__(self, observation: str) -> None:
        super().__init__()
        self.observation = observation

    def reset(
        self, seed: int | None = None, *, row_index: int | None = None
    ) -> tuple[str, dict[str, object]]:
        super().reset(seed, row_index=row_index)
        return self.observation, {}


@pytest.fixture(scope="module")
def chat_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)


def first_prompt(
    tokenizer: PreTrainedTokenizerBase, observation: str, **harness_kwargs: object
) -> str:
    """Decoded first prompt for ``observation`` behind the ``Answer now.`` system prompt."""
    harness = RolloutHarness(
        ObservationClient(observation),
        tokenizer,
        max_turns=1,
        system_prompt="Answer now.",
        **harness_kwargs,
    )
    prompt, _info = harness.reset()
    return tokenizer.decode(prompt["input_ids"][0], skip_special_tokens=False)


def rendered_turns(tokenizer: PreTrainedTokenizerBase, user_content: str) -> str:
    """The system turn and one user turn holding ``user_content``."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Answer now."},
            {"role": "user", "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


class TestRolloutHarnessRepeatSystemPrompt:
    def test_first_user_turn_is_the_observation(
        self, chat_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        observation = "goal\n\npage\n\nQuestion:\ngoal"

        # Act
        prompt = first_prompt(chat_tokenizer, observation)

        # Assert
        assert prompt == rendered_turns(chat_tokenizer, observation)
        assert prompt.count("Answer now.") == 1

    def test_repeat_puts_the_system_prompt_before_the_trailing_question(
        self, chat_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Act
        prompt = first_prompt(
            chat_tokenizer,
            "goal\n\npage\n\nQuestion:\ngoal",
            repeat_system_prompt=True,
        )

        # Assert
        assert prompt == rendered_turns(
            chat_tokenizer, "goal\n\npage\nAnswer now.\n\nQuestion:\ngoal"
        )

    def test_repeat_appends_the_system_prompt_without_a_question(
        self, chat_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Act
        prompt = first_prompt(chat_tokenizer, "page", repeat_system_prompt=True)

        # Assert
        assert prompt == rendered_turns(chat_tokenizer, "page\nAnswer now.")
