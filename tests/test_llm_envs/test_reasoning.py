# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Token-id helpers for a turn's reasoning block."""

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from agilerl.llm_envs.reasoning import (
    REASONING_END_MARKER,
    SUBSEQUENCE_NOT_FOUND,
    find_token_subsequence,
    reasoning_trace_text,
    tokenizer_reasoning_end_ids,
)
from tests import TINY_LLM_FIXTURE_PATH

GEMMA_CHANNEL_OPEN = 100
GEMMA_CHANNEL_CLOSE = 101
QWEN_THINK_CLOSE = 151668
THINK_CLOSE_LT_SLASH = 522
THINK_CLOSE_WORD = 26865
THINK_CLOSE_GT = 29
DOT_LT_SLASH = 3918
GT_BLANK_LINE = 1339
HARMONY_RETURN = 200002
HARMONY_CHANNEL = 200005
HARMONY_START = 200006
HARMONY_END = 200007
HARMONY_MESSAGE = 200008
HARMONY_FINAL = 17196
HARMONY_FINAL_HEADER = (HARMONY_CHANNEL, HARMONY_FINAL, HARMONY_MESSAGE)
PIECES = {
    GEMMA_CHANNEL_OPEN: "<|channel>",
    GEMMA_CHANNEL_CLOSE: "<channel|>",
    QWEN_THINK_CLOSE: "</think>",
    THINK_CLOSE_LT_SLASH: "</",
    THINK_CLOSE_WORD: "think",
    THINK_CLOSE_GT: ">",
    DOT_LT_SLASH: ".</",
    GT_BLANK_LINE: ">\n\n",
    10: "thought\n",
    11: "click('12')",
    12: " ",
    HARMONY_RETURN: "<|return|>",
    HARMONY_CHANNEL: "<|channel|>",
    HARMONY_START: "<|start|>",
    HARMONY_END: "<|end|>",
    HARMONY_MESSAGE: "<|message|>",
    HARMONY_FINAL: "final",
    20: "analysis",
    21: "The ",
    22: " answer is 8.",
    23: "assistant",
    24: "click('8')",
}
SPECIAL_IDS = frozenset(
    {
        GEMMA_CHANNEL_OPEN,
        GEMMA_CHANNEL_CLOSE,
        HARMONY_RETURN,
        HARMONY_CHANNEL,
        HARMONY_START,
        HARMONY_END,
        HARMONY_MESSAGE,
    }
)
HARMONY_ANALYSIS = [HARMONY_CHANNEL, 20, HARMONY_MESSAGE, 21, HARMONY_FINAL, 22]


def decode_pieces(ids: list[int], skip_special_tokens: bool) -> str:
    return "".join(
        PIECES[i] for i in ids if not (skip_special_tokens and i in SPECIAL_IDS)
    )


class HarmonyTokenizer:
    """gpt-oss tokenizer surface: the final-channel header is not one vocab token."""

    unk_token_id = None

    def convert_tokens_to_ids(self, _token: str) -> None:
        return None

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert text == "<|channel|>final<|message|>"
        assert not add_special_tokens
        return list(HARMONY_FINAL_HEADER)


@pytest.fixture(scope="module")
def tiny_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<channel|>"]})
    return tokenizer


class TestTokenizerReasoningEndIds:
    def test_vocab_token_resolves_to_its_single_id(
        self, tiny_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        ids = tokenizer_reasoning_end_ids(tiny_tokenizer, "<channel|>")

        assert ids == (tiny_tokenizer.convert_tokens_to_ids("<channel|>"),)

    def test_marker_outside_the_vocab_resolves_to_its_encoding(
        self, tiny_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        ids = tokenizer_reasoning_end_ids(tiny_tokenizer, REASONING_END_MARKER)

        assert ids == tuple(
            tiny_tokenizer.encode(REASONING_END_MARKER, add_special_tokens=False)
        )
        assert len(ids) > 1

    def test_harmony_final_header_resolves_to_its_encoding(self) -> None:
        ids = tokenizer_reasoning_end_ids(
            HarmonyTokenizer(), "<|channel|>final<|message|>"
        )

        assert ids == HARMONY_FINAL_HEADER


class TestFindTokenSubsequence:
    def test_single_token_pattern_is_found(self) -> None:
        assert find_token_subsequence([1, 2, 7, 3], [7]) == 2

    def test_multi_token_pattern_is_found(self) -> None:
        assert find_token_subsequence([1, 7, 9, 7, 8, 3], [7, 8]) == 3

    def test_pattern_at_position_zero_is_found(self) -> None:
        assert find_token_subsequence([7, 8, 1], [7, 8]) == 0

    def test_first_of_several_occurrences_wins(self) -> None:
        assert find_token_subsequence([1, 7, 2, 7], [7]) == 1

    def test_absent_pattern_reports_not_found(self) -> None:
        assert find_token_subsequence([1, 2, 3], [7]) == SUBSEQUENCE_NOT_FOUND

    def test_pattern_longer_than_the_sequence_reports_not_found(self) -> None:
        assert find_token_subsequence([7], [7, 8]) == SUBSEQUENCE_NOT_FOUND

    def test_empty_pattern_never_matches(self) -> None:
        assert find_token_subsequence([1, 2, 3], []) == SUBSEQUENCE_NOT_FOUND

    def test_truncated_trailing_pattern_reports_not_found(self) -> None:
        assert find_token_subsequence([1, 2, 7], [7, 8]) == SUBSEQUENCE_NOT_FOUND

    def test_partial_match_at_the_last_start_reports_not_found(self) -> None:
        assert find_token_subsequence([1, 7, 9], [7, 8]) == SUBSEQUENCE_NOT_FOUND


class TestReasoningTraceText:
    def test_keeps_harmony_channel_headers_and_drops_the_return(self) -> None:
        # Arrange
        gen_ids = [
            *HARMONY_ANALYSIS,
            HARMONY_END,
            HARMONY_START,
            23,
            *HARMONY_FINAL_HEADER,
            24,
            HARMONY_RETURN,
        ]

        # Act
        trace = reasoning_trace_text(gen_ids, SPECIAL_IDS, decode_pieces)

        # Assert
        assert trace == (
            "<|channel|>analysis<|message|>The final answer is 8.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>click('8')"
        )

    def test_keeps_gemma_channel_markers(self) -> None:
        trace = reasoning_trace_text(
            [GEMMA_CHANNEL_OPEN, 10, GEMMA_CHANNEL_CLOSE, 11],
            SPECIAL_IDS,
            decode_pieces,
        )

        assert trace == "<|channel>thought\n<channel|>click('12')"

    def test_all_special_turn_is_empty(self) -> None:
        assert reasoning_trace_text([HARMONY_RETURN], SPECIAL_IDS, decode_pieces) == ""
