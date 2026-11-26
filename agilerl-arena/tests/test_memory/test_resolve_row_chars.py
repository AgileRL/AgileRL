# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""A dataset's characters per trained row as a model family's tokens."""

import pytest

from agilerl.arena.memory.resolve.row_chars import (
    DEFAULT_TOKENIZER_FIT,
    TOKENIZER_FITS,
    RowChars,
    row_tokens_from_chars,
)
from agilerl.arena.memory.resolve.split import RowTokens
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO


class TestRowTokensFromChars:
    def test_every_catalog_family_has_a_fit(self) -> None:
        # Arrange
        families = {info.model_type for info in SUPPORTED_MODEL_INFO.values()}

        # Assert
        assert families <= TOKENIZER_FITS.keys()

    def test_empty_rows_are_the_template_alone(self) -> None:
        # Arrange
        overhead = TOKENIZER_FITS["qwen3_5"].template_tokens

        # Act
        tokens = row_tokens_from_chars(RowChars(0.0, 0.0), "qwen3_5")

        # Assert
        assert tokens == RowTokens(overhead, overhead**2)

    def test_rows_of_one_length_keep_one_length(self) -> None:
        # Arrange
        ratio, overhead = TOKENIZER_FITS["gpt_oss"]

        # Act
        tokens = row_tokens_from_chars(RowChars(1000.0, 1000.0**2), "gpt_oss")

        # Assert: every row is 1000 / r + o tokens.
        assert tokens.mean == pytest.approx(1000.0 / ratio + overhead)
        assert tokens.mean_square == pytest.approx(tokens.mean**2)

    def test_the_spread_scales_by_characters_per_token(self) -> None:
        # Arrange: rows of 500 and 1500 characters.
        chars = RowChars(1000.0, (500.0**2 + 1500.0**2) / 2)
        ratio = TOKENIZER_FITS["gemma4"].chars_per_token

        # Act
        tokens = row_tokens_from_chars(chars, "gemma4")

        # Assert: the template shifts every row, so only the variance's scale changes.
        variance = tokens.mean_square - tokens.mean**2
        assert variance == pytest.approx((chars.mean_square - chars.mean**2) / ratio**2)

    def test_an_unknown_family_takes_the_default_fit(self) -> None:
        # Arrange
        ratio, overhead = DEFAULT_TOKENIZER_FIT

        # Act
        tokens = row_tokens_from_chars(RowChars(380.0, 380.0**2), "llama")

        # Assert
        assert tokens.mean == pytest.approx(380.0 / ratio + overhead)
        assert tokens.mean_square == pytest.approx(tokens.mean**2)

    @pytest.mark.parametrize(
        "chars",
        [
            pytest.param(RowChars(-1.0, 1.0), id="mean"),
            pytest.param(RowChars(1.0, -1.0), id="mean-square"),
        ],
    )
    def test_negative_stats_are_refused(self, chars: RowChars) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="must not be negative"):
            row_tokens_from_chars(chars, "qwen3_5")
