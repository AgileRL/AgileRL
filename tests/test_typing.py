# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from agilerl.typing import coerce_action_mask, numpy_action_mask


class TestCoerceActionMask:
    def test_none_passes_through(self):
        assert coerce_action_mask(None) is None

    def test_ndarray_passes_through(self):
        arr = np.array([1, 0, 1])
        assert coerce_action_mask(arr) is arr

    def test_numeric_sequence_is_coerced_to_list(self):
        assert coerce_action_mask((1, 0, True)) == [1, 0, True]

    def test_sequence_with_non_numeric_returns_none(self):
        assert coerce_action_mask([1, "x", 0]) is None

    def test_str_and_bytes_are_not_treated_as_sequences(self):
        assert coerce_action_mask("101") is None
        assert coerce_action_mask(b"101") is None

    def test_unsupported_type_returns_none(self):
        assert coerce_action_mask(3.5) is None


class TestNumpyActionMask:
    def test_tensor_is_converted_to_numpy(self):
        out = numpy_action_mask(torch.tensor([1, 0, 1]))
        assert isinstance(out, np.ndarray)
        assert out.tolist() == [1, 0, 1]

    def test_plain_ndarray_passes_through(self):
        out = numpy_action_mask(np.array([[1, 0], [0, 1]]))
        assert isinstance(out, np.ndarray)
        assert out.shape == (2, 2)

    def test_object_dtype_array_is_stacked(self):
        ragged = np.empty(2, dtype=object)
        ragged[0] = np.array([1, 0])
        ragged[1] = np.array([0, 1])
        out = numpy_action_mask(ragged)
        assert out.shape == (2, 2)

    def test_sequence_of_arrays_is_stacked(self):
        out = numpy_action_mask([np.array([1, 0]), np.array([0, 1])])
        assert out.shape == (2, 2)
