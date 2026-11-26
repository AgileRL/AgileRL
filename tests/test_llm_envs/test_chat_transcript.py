# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

pytest.importorskip("transformers", reason="LLM dependencies not installed")

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from agilerl.llm_envs.chat_transcript import (
    chat_messages,
    encode_chat_transcript,
    render_chat_prompt,
)
from tests import TINY_LLM_FIXTURE_PATH


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)


def request_messages() -> list[dict[str, Any]]:
    """Messages as an OpenAI client sends them: a text-part user turn after a past exchange."""
    return [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "tool_calls": None},
        {"role": "user", "content": [{"type": "text", "text": "again"}]},
    ]


class TestRenderChatPrompt:
    def test_is_the_context_sft_trains_the_next_turn_after(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        messages = request_messages()
        reply = {"role": "assistant", "content": "hello again"}

        # Act
        text = render_chat_prompt(tokenizer, messages, None, None)

        # Assert
        ids, mask = encode_chat_transcript(
            tokenizer,
            chat_messages([*messages, reply]),
            None,
            None,
            trained_from=len(messages),
        )
        assert ids[: mask.index(True)] == tokenizer.encode(
            text, add_special_tokens=False
        )
        assert text.count("again") == 1

    def test_json_text_renders_like_a_list(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        messages = request_messages()

        text = render_chat_prompt(tokenizer, json.dumps(messages), None, None)

        assert text == render_chat_prompt(tokenizer, messages, None, None)

    def test_empty_tools_render_like_none(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        messages = request_messages()

        text = render_chat_prompt(tokenizer, messages, [], None)

        assert text == render_chat_prompt(tokenizer, messages, None, None)

    def test_passes_chat_template_kwargs_to_the_template(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        greeting = copy.deepcopy(tokenizer)
        greeting.chat_template = (
            "{{ greeting }}{% for message in messages %}{{ message.content }}"
            "{% endfor %}{% if add_generation_prompt %}>{% endif %}"
        )

        # Act
        text = render_chat_prompt(
            greeting, request_messages()[:2], None, None, {"greeting": "hey "}
        )

        # Assert
        assert text == "hey Be brief.hi>"

    def test_rejects_a_prompt_ending_with_an_assistant_turn(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        messages = request_messages()[:3]

        with pytest.raises(ValueError, match="must not end with an assistant turn"):
            render_chat_prompt(tokenizer, messages, None, None)
