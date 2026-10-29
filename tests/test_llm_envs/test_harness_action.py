# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Text after the first think close, and where the system prompt sits."""

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from agilerl.llm_envs.chat_formats import action_span
from agilerl.llm_envs.harness import RolloutHarness
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


GENERATED_ID = 9


class RecordingTokenizer(MiniTokenizer):
    """Decodes ``GENERATED_ID`` as ``text`` and records each prompt it tokenizes."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def decode(self, ids: list[int], **_: object) -> str:
        return self.text if list(ids) == [GENERATED_ID] else ""

    def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
        self.prompts.append(texts[0])
        ids = torch.tensor([[1]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class PieceTokenizer(RecordingTokenizer):
    """Decodes ids piece by piece, dropping ``special_ids`` when asked to skip them."""

    def __init__(self, pieces: dict[int, str], special_ids: frozenset[int]) -> None:
        super().__init__("")
        self.pieces = pieces
        self.special_ids = special_ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(
            self.pieces[i]
            for i in ids
            if not (skip_special_tokens and i in self.special_ids)
        )

    @property
    def all_special_ids(self) -> list[int]:
        return sorted(self.special_ids)

    def get_vocab(self) -> dict[str, int]:
        return {piece: token_id for token_id, piece in self.pieces.items()}

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.get_vocab().get(token)

    def encode(self, text: str, **kwargs: object) -> list[int]:
        vocab = self.get_vocab()
        return [vocab[text]] if text in vocab else super().encode(text, **kwargs)


GEMMA_CHANNEL_OPEN = 100
GEMMA_CHANNEL_CLOSE = 101
QWEN_THINK_CLOSE = 151668
REASONING_ID = 10
ACTION_ID = 11
GEMMA_TOOL_PIECES = {
    102: "<|tool_call>",
    103: "<tool_call|>",
    104: "<|tool_response>",
    105: "<tool_response|>",
}
GEMMA_TOKENIZER_PIECES = {
    **GEMMA_TOOL_PIECES,
    GEMMA_CHANNEL_OPEN: "<|channel>",
    GEMMA_CHANNEL_CLOSE: "<channel|>",
    REASONING_ID: "thought\nthe lamp is $5",
    ACTION_ID: "click('12')",
}
QWEN_TOKENIZER_PIECES = {
    QWEN_THINK_CLOSE: "</think>",
    REASONING_ID: "the lamp is $5",
    ACTION_ID: "\n\nclick('12')",
}
HARMONY_CHANNEL = 200005
HARMONY_FINAL = 17196
HARMONY_MESSAGE = 200008
HARMONY_END = 200007
HARMONY_START = 200006
HARMONY_ANALYSIS = 20
HARMONY_ASSISTANT = 21
HARMONY_CALL = 200012
HARMONY_TOKENIZER_PIECES = {
    HARMONY_CALL: "<|call|>",
    HARMONY_CHANNEL: "<|channel|>",
    HARMONY_FINAL: "final",
    HARMONY_MESSAGE: "<|message|>",
    HARMONY_END: "<|end|>",
    HARMONY_START: "<|start|>",
    HARMONY_ANALYSIS: "analysis",
    HARMONY_ASSISTANT: "assistant",
    REASONING_ID: "the final lamp is $5",
    ACTION_ID: "click('12')",
}
HARMONY_SPECIAL_IDS = frozenset(
    {HARMONY_CHANNEL, HARMONY_MESSAGE, HARMONY_END, HARMONY_START}
)
HARMONY_FINAL_HEADER = (HARMONY_CHANNEL, HARMONY_FINAL, HARMONY_MESSAGE)


def rewrite_action(text: str) -> str:
    return f"quoted({text})"


def step_once(
    tokenizer: RecordingTokenizer,
    gen_ids: list[int],
    processor: object = None,
    chat_format: str | None = None,
    include_reasoning: bool = False,
) -> tuple[list[str], str]:
    """Step one generation and return the env's actions and the restarted prompt."""
    client = RecordingClient()
    harness = RolloutHarness(
        client,
        tokenizer,
        max_turns=2,
        apply_chat_template=False,
        segment_prompt_tokens=2,
        action_processor=processor,
        chat_format=chat_format,
        include_reasoning=include_reasoning,
    )
    harness.reset()
    harness.step(torch.tensor([[1, *gen_ids]], dtype=torch.long))
    return client.actions, tokenizer.prompts[-1]


def restarted_prompt(action: str, *, processor: object = None) -> tuple[list[str], str]:
    """Step one action and return the env's actions and the restarted prompt."""
    return step_once(RecordingTokenizer(action), [GENERATED_ID], processor)


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


class TestRolloutHarnessReasoningEnd:
    def test_special_token_end_marker_keeps_reasoning_out_of_the_action(self) -> None:
        # Arrange
        tokenizer = PieceTokenizer(
            GEMMA_TOKENIZER_PIECES,
            frozenset({GEMMA_CHANNEL_OPEN, GEMMA_CHANNEL_CLOSE}),
        )
        gen_ids = [GEMMA_CHANNEL_OPEN, REASONING_ID, GEMMA_CHANNEL_CLOSE, ACTION_ID]

        # Act
        actions, prompt = step_once(tokenizer, gen_ids, chat_format="gemma4")

        # Assert
        assert actions == ["click('12')"]
        assert "1. click('12')" in prompt
        assert "the lamp is $5" not in prompt

    def test_plain_text_think_close_splits_without_a_chat_format(self) -> None:
        # Arrange
        tokenizer = PieceTokenizer(QWEN_TOKENIZER_PIECES, frozenset())

        # Act
        actions, prompt = step_once(
            tokenizer, [REASONING_ID, QWEN_THINK_CLOSE, ACTION_ID]
        )

        # Assert
        assert actions == ["click('12')"]
        assert "1. click('12')" in prompt
        assert "the lamp is $5" not in prompt

    def test_unclosed_reasoning_is_sent_whole(self) -> None:
        tokenizer = PieceTokenizer(QWEN_TOKENIZER_PIECES, frozenset())

        actions, _prompt = step_once(tokenizer, [REASONING_ID, ACTION_ID])

        assert actions == ["the lamp is $5\n\nclick('12')"]

    def test_harmony_final_channel_keeps_the_analysis_out_of_the_action(
        self,
    ) -> None:
        # Arrange
        tokenizer = PieceTokenizer(HARMONY_TOKENIZER_PIECES, HARMONY_SPECIAL_IDS)
        gen_ids = [
            HARMONY_CHANNEL,
            HARMONY_ANALYSIS,
            HARMONY_MESSAGE,
            REASONING_ID,
            HARMONY_END,
            HARMONY_START,
            HARMONY_ASSISTANT,
            *HARMONY_FINAL_HEADER,
            ACTION_ID,
        ]

        # Act
        actions, prompt = step_once(tokenizer, gen_ids, chat_format="harmony")

        # Assert
        assert actions == ["click('12')"]
        assert "1. click('12')" in prompt
        assert "the final lamp is $5" not in prompt


GEMMA_TURN_END = 106
GEMMA_TRACE_PIECES = {**GEMMA_TOKENIZER_PIECES, GEMMA_TURN_END: "<turn|>"}
GEMMA_TRACE_SPECIAL_IDS = frozenset(
    {GEMMA_CHANNEL_OPEN, GEMMA_CHANNEL_CLOSE, GEMMA_TURN_END}
)


class TestRolloutHarnessIncludeReasoning:
    def test_sends_the_whole_turn_with_its_markers(self) -> None:
        # Arrange
        tokenizer = PieceTokenizer(GEMMA_TRACE_PIECES, GEMMA_TRACE_SPECIAL_IDS)
        gen_ids = [
            GEMMA_CHANNEL_OPEN,
            REASONING_ID,
            GEMMA_CHANNEL_CLOSE,
            ACTION_ID,
            GEMMA_TURN_END,
        ]

        # Act
        actions, prompt = step_once(
            tokenizer,
            gen_ids,
            chat_format="gemma4",
            include_reasoning=True,
        )

        # Assert
        assert actions == ["<|channel>thought\nthe lamp is $5<channel|>click('12')"]
        assert "1. click('12')" in prompt
        assert "the lamp is $5" not in prompt

    def test_think_trace_reaches_a_format_reward(self) -> None:
        tokenizer = PieceTokenizer(QWEN_TOKENIZER_PIECES, frozenset())

        actions, _prompt = step_once(
            tokenizer,
            [REASONING_ID, QWEN_THINK_CLOSE, ACTION_ID],
            include_reasoning=True,
        )

        assert actions == ["the lamp is $5</think>\n\nclick('12')"]

    def test_default_sends_only_the_action(self) -> None:
        tokenizer = PieceTokenizer(GEMMA_TRACE_PIECES, GEMMA_TRACE_SPECIAL_IDS)

        actions, _prompt = step_once(
            tokenizer,
            [GEMMA_CHANNEL_OPEN, REASONING_ID, GEMMA_CHANNEL_CLOSE, ACTION_ID],
            chat_format="gemma4",
        )

        assert actions == ["click('12')"]


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
