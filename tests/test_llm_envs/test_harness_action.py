# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Text after the first think close."""

import torch

from agilerl.llm_envs.harness import RolloutHarness, action_span
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


class TestTrailingInstruction:
    def test_leaves_the_question_after_the_instruction(self) -> None:
        harness = RolloutHarness.__new__(RolloutHarness)
        harness._system_prompt = "Answer now."

        rendered = harness._with_trailing_instruction("goal\n\npage\n\nQuestion:\ngoal")

        assert rendered == "goal\n\npage\nAnswer now.\n\nQuestion:\ngoal"
