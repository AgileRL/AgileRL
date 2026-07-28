"""Guards the runtime array-annotation checker installed in ``tests/conftest.py``."""

import os

import numpy as np
import pytest
from jaxtyping import TypeCheckError

from agilerl.components.segment_tree import SumSegmentTree
from agilerl.utils.algo_utils import is_str_keyed_dict

HOOK_DISABLED = bool(os.environ.get("AGILERL_NO_TYPE_HOOK"))
requires_hook = pytest.mark.skipif(
    HOOK_DISABLED, reason="runtime type hook disabled via AGILERL_NO_TYPE_HOOK"
)


@requires_hook
def test_hook_rejects_wrong_dtype():
    with pytest.raises(TypeCheckError):
        SumSegmentTree(8).get_batch(np.array([0.0, 1.0]))


@requires_hook
def test_hook_rejects_wrong_rank():
    with pytest.raises(TypeCheckError):
        SumSegmentTree(8).update_batch(np.array([[0, 1]]), np.array([[1.0, 2.0]]))


@requires_hook
def test_hook_rejects_disagreeing_axis_sizes():
    """A repeated axis name must bind to one size across arguments."""
    with pytest.raises(TypeCheckError):
        SumSegmentTree(8).update_batch(np.array([0, 1, 2]), np.array([1.0, 2.0]))


@requires_hook
def test_hook_accepts_a_correct_call():
    SumSegmentTree(8).update_batch(np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]))


@requires_hook
def test_hook_ignores_non_array_annotations():
    """Only array hints are checked, so the rest of the annotation surface is free."""
    assert is_str_keyed_dict(12345) is False
