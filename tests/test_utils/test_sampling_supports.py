# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for reading vLLM sampling supports."""

from types import SimpleNamespace

import pytest
import torch

from agilerl.utils.sampling_supports import (
    concat_sampling_supports,
    extract_vllm_sampling_supports,
)


class TestExtractVllmSamplingSupports:
    def test_pads_each_token_support_to_the_widest(self) -> None:
        # Arrange
        output = SimpleNamespace(
            token_ids=[4, 7, 2],
            sampling_mask=SimpleNamespace(token_ids=[[4, 9, 11], [7], [2, 3]]),
        )

        # Act
        supports = extract_vllm_sampling_supports(output)

        # Assert
        assert supports.dtype == torch.int32
        assert supports.tolist() == [[4, 9, 11], [7, -1, -1], [2, 3, -1]]

    def test_an_empty_generation_has_no_rows(self) -> None:
        output = SimpleNamespace(
            token_ids=[], sampling_mask=SimpleNamespace(token_ids=[])
        )

        assert extract_vllm_sampling_supports(output).shape == (0, 1)

    def test_a_missing_sampling_mask_raises(self) -> None:
        output = SimpleNamespace(token_ids=[1, 2], sampling_mask=None)

        with pytest.raises(RuntimeError, match="for None of 2 generated tokens"):
            extract_vllm_sampling_supports(output)

    def test_a_mask_short_of_the_tokens_raises(self) -> None:
        output = SimpleNamespace(
            token_ids=[1, 2], sampling_mask=SimpleNamespace(token_ids=[[1]])
        )

        with pytest.raises(RuntimeError, match="for 1 of 2 generated tokens"):
            extract_vllm_sampling_supports(output)


class TestConcatSamplingSupports:
    def test_stacks_generations_at_the_widest_support(self) -> None:
        # Arrange
        turn = torch.tensor([[1, 2], [3, -1]], dtype=torch.int32)
        continuation = torch.tensor([[5, 6, 7]], dtype=torch.int32)

        # Act
        supports = concat_sampling_supports([turn, continuation])

        # Assert
        assert supports.tolist() == [[1, 2, -1], [3, -1, -1], [5, 6, 7]]

    def test_no_generations_give_no_rows(self) -> None:
        assert concat_sampling_supports([]).shape == (0, 1)
